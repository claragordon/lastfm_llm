const searchInput = document.getElementById('artist-search');
const resultsEl = document.getElementById('search-results');
const historyEl = document.getElementById('history-chips');
const predictBtn = document.getElementById('predict-btn');
const clearBtn = document.getElementById('clear-btn');
const topKInput = document.getElementById('top-k');
const predictionsBody = document.getElementById('predictions-body');
const unknownWarning = document.getElementById('unknown-warning');
const showAttentionInput = document.getElementById('show-attention');
const attentionPanel = document.getElementById('attention-panel');
const attentionContent = document.getElementById('attention-content');

let selectedArtists = [];
let searchDebounce = null;
const imageCache = new Map();
const imageInFlight = new Set();
const imageRetryAfterMs = 30000;
const imageLastRetryTs = new Map();

function updateUrlFromState() {
  const url = new URL(window.location.href);
  const ids = selectedArtists.map((a) => a.artist_id).filter(Boolean);
  if (ids.length > 0) {
    url.searchParams.set('seq', ids.join(','));
  } else {
    url.searchParams.delete('seq');
  }
  window.history.replaceState({}, '', `${url.pathname}${url.search}`);
}

async function hydrateArtistsFromIds(ids) {
  const artists = [];
  for (const artistId of ids) {
    try {
      const res = await fetch(`/artist_media?artist_id=${encodeURIComponent(artistId)}`);
      if (!res.ok) continue;
      const data = await res.json();
      artists.push({
        artist_id: artistId,
        artist_name: data.artist_name || artistId,
        artist_image_url: data.artist_image_url || null,
      });
      imageCache.set(artistId, data.artist_image_url || null);
    } catch (err) {
      console.error(err);
    }
  }
  return artists;
}

async function restoreStateFromUrl() {
  const url = new URL(window.location.href);
  const seqRaw = url.searchParams.get('seq');
  if (!seqRaw) {
    renderHistory();
    return;
  }
  const ids = seqRaw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (ids.length === 0) {
    renderHistory();
    return;
  }
  selectedArtists = await hydrateArtistsFromIds(ids);
  renderHistory();
}

function renderAttention(attention) {
  if (!showAttentionInput.checked || !attention) {
    attentionPanel.hidden = true;
    attentionContent.innerHTML = '';
    return;
  }
  attentionPanel.hidden = false;
  const heads = attention.per_head || [];
  const names = attention.history_artist_names || [];
  if (!heads.length || !names.length) {
    attentionContent.innerHTML = '<div>No attention data available.</div>';
    return;
  }

  const colorFor = (weight, rowMax) => {
    const t = rowMax > 0 ? weight / rowMax : 0;
    // Yellow (low attention) to red (high attention).
    const hue = 55 - (55 * t);
    const lightness = 84 - (34 * t);
    return `hsl(${hue} 95% ${lightness}%)`;
  };

  const headerCells = names.map((name, idx) => `<th title="Position ${idx}">${name}</th>`).join('');
  const rows = heads.map((head) => {
    const weights = head.weights || [];
    const rowMax = Math.max(...weights, 0);
    const cells = names.map((name, idx) => {
      const w = Number(weights[idx] || 0);
      return `
        <td
          class="attention-cell"
          style="background:${colorFor(w, rowMax)}"
          title="Head ${head.head_index} · Pos ${idx} · ${name} · weight=${w.toFixed(4)}"
        >
          ${w.toFixed(2)}
        </td>
      `;
    }).join('');
    return `<tr><th>Head ${head.head_index}</th>${cells}</tr>`;
  }).join('');

  attentionContent.innerHTML = `
    <div class="attention-legend">Color scale: yellow = low attention, red = high attention (per head).</div>
    <table class="attention-heatmap">
      <thead>
        <tr>
          <th>Head \\ Pos</th>
          ${headerCells}
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}

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
      updateUrlFromState();
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
      updateUrlFromState();
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
    include_attention: showAttentionInput.checked,
    attention_top_n: 5,
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
    updateUrlFromState();
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
  renderAttention(data.attention);
});

clearBtn.addEventListener('click', () => {
  selectedArtists = [];
  predictionsBody.innerHTML = '';
  unknownWarning.textContent = '';
  renderAttention(null);
  updateUrlFromState();
  renderHistory();
});

showAttentionInput.addEventListener('change', () => {
  if (!showAttentionInput.checked) {
    renderAttention(null);
  }
});

restoreStateFromUrl();