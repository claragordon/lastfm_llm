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

function renderHistory() {
  historyEl.innerHTML = '';
  if (selectedArtists.length === 0) {
    historyEl.innerHTML = '<span class="chip">No artists selected yet</span>';
    return;
  }
  selectedArtists.forEach((artistId, idx) => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = `${idx + 1}. ${artistId} ×`;
    chip.onclick = () => {
      selectedArtists.splice(idx, 1);
      renderHistory();
    };
    historyEl.appendChild(chip);
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
    row.textContent = `${item.artist_id}  (token ${item.token_id})`;
    row.onclick = () => {
      selectedArtists.push(item.artist_id);
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
    history_artist_ids: selectedArtists,
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
    unknownWarning.textContent = `Ignored unknown ids: ${data.unknown_artist_ids.join(', ')}`;
  }

  data.predictions.forEach((p, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${p.artist_id}</td>
      <td>${p.token_id}</td>
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
