const searchInput = document.getElementById('artist-search');
const resultsEl = document.getElementById('search-results');
const historyEl = document.getElementById('history-chips');
const predictBtn = document.getElementById('predict-btn');
const clearBtn = document.getElementById('clear-btn');
const topKInput = document.getElementById('top-k');
const predictionsBody = document.getElementById('predictions-body');
const unknownWarning = document.getElementById('unknown-warning');

let selectedArtists = [];
let searchDebounce = null;
const imageCache = new Map();
const imageInFlight = new Set();
const imageRetryAfterMs = 30000;
const imageLastRetryTs = new Map();

function avatarHtml(url, label) {
  if (url) {
    return `<img class="artist-avatar" src="${url}" alt="${label}" loading="lazy" />`;
  }
  return '<span class="artist-avatar artist-avatar--placeholder"></span>';
}

async function ensureArtistImage(artistId) {
  if (!artistId || imageInFlight.has(artistId)) return;
  const cached = imageCache.get(artistId);
  if (cached) return; // already have a real URL
  const now = Date.now();
  if ((imageLastRetryTs.get(artistId) || 0) + imageRetryAfterMs > now) return;

  imageInFlight.add(artistId);
  imageLastRetryTs.set(artistId, now);
  try {
    const res = await fetch(`/artist_media?artist_id=${encodeURIComponent(artistId)}`);
    if (!res.ok) return;
    const data = await res.json();
    imageCache.set(artistId, data.artist_image_url || null);
    selectedArtists = selectedArtists.map((a) => (
      a.artist_id === artistId
        ? { ...a, artist_image_url: data.artist_image_url || null }
        : a
    ));
    renderHistory();
  } catch (err) {
    console.error(err);
  } finally {
    imageInFlight.delete(artistId);
  }
}

function renderHistory() {
  historyEl.innerHTML = '';
  if (selectedArtists.length === 0) {
    historyEl.innerHTML = '<span class="chip">No artists selected yet</span>';
    return;
  }
  selectedArtists.forEach((artist, idx) => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    const imageUrl = artist.artist_image_url ?? imageCache.get(artist.artist_id) ?? null;
    chip.innerHTML = `${avatarHtml(imageUrl, artist.artist_name)}<span>${idx + 1}. ${artist.artist_name} ×</span>`;
    chip.onclick = () => {
      selectedArtists.splice(idx, 1);
      renderHistory();
    };
    historyEl.appendChild(chip);
    if (!imageUrl) {
      ensureArtistImage(artist.artist_id);
    }
  });
}

function renderSearchResults(items) {
  resultsEl.innerHTML = '';
  if (!items.length) {
    return;
  }
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'result-item';
    const imageUrl = item.artist_image_url ?? imageCache.get(item.artist_id) ?? null;
    if (item.artist_image_url !== undefined) {
      imageCache.set(item.artist_id, item.artist_image_url);
    }
    row.innerHTML = `${avatarHtml(imageUrl, item.artist_name)}<span>${item.artist_name}</span>`;
    row.onclick = () => {
      selectedArtists.push({
        artist_id: item.artist_id,
        artist_name: item.artist_name,
        artist_image_url: imageUrl,
      });
      searchInput.value = '';
      resultsEl.innerHTML = '';
      renderHistory();
    };
    resultsEl.appendChild(row);
  });
}

async function queryArtists(query) {
  const url = `/artists?q=${encodeURIComponent(query)}&limit=20`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error('Artist search failed');
  }
  const data = await res.json();
  renderSearchResults(data.results || []);
}

searchInput.addEventListener('input', () => {
  const q = searchInput.value.trim();
  if (searchDebounce) {
    clearTimeout(searchDebounce);
  }
  searchDebounce = setTimeout(() => {
    queryArtists(q).catch((err) => {
      console.error(err);
      resultsEl.innerHTML = '';
    });
  }, 150);
});

predictBtn.addEventListener('click', async () => {
  predictionsBody.innerHTML = '';
  unknownWarning.textContent = '';

  const topK = Number(topKInput.value || 10);
  const payload = {
    history_artist_ids: selectedArtists.map((a) => a.artist_id),
    top_k: topK,
  };

  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) {
    unknownWarning.textContent = data.detail || 'Prediction failed';
    return;
  }

  if (data.unknown_artist_ids && data.unknown_artist_ids.length > 0) {
    unknownWarning.textContent = 'Some unrecognized artists were ignored.';
  }

  if (data.history_used_artist_ids && data.history_used_artist_names) {
    selectedArtists = data.history_used_artist_ids.map((artistId, i) => ({
      artist_id: artistId,
      artist_name: data.history_used_artist_names[i] || artistId,
      artist_image_url: (data.history_used_artist_images || [])[i] || null,
    }));
    selectedArtists.forEach((artist) => {
      if (artist.artist_image_url !== undefined) {
        imageCache.set(artist.artist_id, artist.artist_image_url);
      }
    });
    renderHistory();
  }

  data.predictions.forEach((p, idx) => {
    const tr = document.createElement('tr');
    const imageUrl = p.artist_image_url ?? imageCache.get(p.artist_id) ?? null;
    if (p.artist_image_url !== undefined) {
      imageCache.set(p.artist_id, p.artist_image_url);
    }
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td><span class="artist-cell">${avatarHtml(imageUrl, p.artist_name)}<span>${p.artist_name}</span></span></td>
      <td>${p.prob.toFixed(6)}</td>
      <td>${p.logit.toFixed(4)}</td>
    `;
    predictionsBody.appendChild(tr);
  });
});

clearBtn.addEventListener('click', () => {
  selectedArtists = [];
  predictionsBody.innerHTML = '';
  unknownWarning.textContent = '';
  renderHistory();
});

renderHistory();