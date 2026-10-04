// Methods shown in the UI. `key` is the results.json key, `cls` the CSS/API key.
const METHODS = [
    { key: 'greedy',            cls: 'greedy',            name: 'Greedy',              type: 'Popularity-based' },
    { key: 'content_filtering', cls: 'content_filtering', name: 'Content Filtering',   type: 'Cosine similarity' },
    { key: 'mmr_0.7',           cls: 'mmr',               name: 'MMR (λ=0.7)',         type: 'Similarity + redundancy penalty' },
    { key: 'mmr_floor',         cls: 'mmr_floor',         name: 'MMR + niche floor',   type: 'MMR + ≥20% niche floor' },
    { key: 'graph_dpp_rerank',  cls: 'graph_dpp_rerank',  name: 'Graph DPP Rerank',    type: 'Similarity + DPP + niche floor' },
];
const METRICS = [
    { key: 'ild',            label: 'ILD (Diversity)',    digits: 3, better: 'high' },
    { key: 'gini',           label: 'Gini (popularity concentration)', digits: 3, better: null },
    { key: 'avg_popularity', label: 'Avg popularity',     digits: 1, better: null },
    { key: 'niche_pct',      label: 'Niche songs (%)',    digits: 1, better: 'high' },
];

function esc(value) {
    const d = document.createElement('div');
    d.textContent = value == null ? '' : String(value);
    return d.innerHTML;
}

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    animateNumbers();
    loadResultsData();
    initScrollSpy();
    setupSearch();
});

// Load output/results.json (written by run_pipeline.py) and render the dashboard sections
async function loadResultsData() {
    if (!document.getElementById('metrics-dashboard')) return; // not the dashboard page
    try {
        const response = await fetch('output/results.json');
        if (!response.ok) throw new Error('HTTP ' + response.status);
        const data = await response.json();
        renderMetrics(data);
        renderComparison(data);
        renderPaired(data);
    } catch (e) {
        console.error('Could not load output/results.json:', e);
        const msg = '<div class="load-error">Results not found. Run <code>python run_pipeline.py --skip-dl --plots</code> ' +
                    'and open this page through <code>python server.py</code>.</div>';
        document.getElementById('metrics-dashboard').innerHTML = msg;
        ['comparison-table', 'paired-table'].forEach(id => {
            const t = document.getElementById(id);
            if (t) t.innerHTML = '<tbody><tr><td>' + msg + '</td></tr></tbody>';
        });
    }
    animateBarsOnScroll();
}

function fmt(v, digits) { return Number(v).toFixed(digits); }

function bestKey(data, metric) {
    const vals = METHODS.filter(m => data[m.key]).map(m => [m.key, data[m.key][metric.key]]);
    if (!metric.better) return null;
    return vals.reduce((a, b) => ((metric.better === 'high') === (b[1] > a[1]) ? b : a))[0];
}

function renderMetrics(data) {
    const first = data[METHODS[0].key];
    document.getElementById('metrics-desc').textContent =
        `Mean over ${first.n_runs} paired runs (K=${first.K}, RNG seed ${first.rng_seed}) with random seed songs`;

    // Bar widths are relative to the largest mean of that metric so bars are comparable
    const maxOf = {};
    METRICS.forEach(m => {
        maxOf[m.key] = Math.max(...METHODS.filter(x => data[x.key]).map(x => data[x.key][m.key]), 1e-9);
    });

    document.getElementById('metrics-dashboard').innerHTML = METHODS.filter(m => data[m.key]).map(m => {
        const r = data[m.key];
        const items = METRICS.map(mt => {
            const width = Math.max(2, Math.round(100 * r[mt.key] / maxOf[mt.key]));
            return `
                <div class="metric-item">
                    <div class="metric-bar-container">
                        <div class="metric-bar metric-bar-${m.cls}" style="--bar-width: ${width}%"></div>
                    </div>
                    <div class="metric-info">
                        <span class="metric-label">${esc(mt.label)}</span>
                        <span class="metric-value">${fmt(r[mt.key], mt.digits)}
                            <span class="metric-sub">± ${fmt(r[mt.key + '_std'], mt.digits)} · median ${fmt(r[mt.key + '_median'], mt.digits)}</span>
                        </span>
                    </div>
                </div>`;
        }).join('');
        return `
            <div class="metric-column" id="metric-${m.cls}">
                <div class="metric-header metric-header-${m.cls}">
                    <h3>${esc(m.name)}</h3>
                    <span>${esc(m.type)}</span>
                </div>
                <div class="metric-body">${items}</div>
            </div>`;
    }).join('');

    document.getElementById('metrics-note').textContent =
        'ILD and niche % are bimodal (the standard deviation is as large as the mean for ILD), so medians are shown ' +
        'alongside means. Bar lengths are relative to the largest value of each metric.';
}

function renderComparison(data) {
    const ms = METHODS.filter(m => data[m.key]);
    const head = `<thead><tr><th>Metric</th>${ms.map(m => `<th class="th-${m.cls}">${esc(m.name)}</th>`).join('')}<th>Leader by mean*</th></tr></thead>`;

    const rows = METRICS.map(mt => {
        const best = bestKey(data, mt);
        const cells = ms.map(m => {
            const r = data[m.key];
            return `<td class="${m.key === best ? 'td-best' : ''}">${fmt(r[mt.key], mt.digits)}<span class="td-note">median ${fmt(r[mt.key + '_median'], mt.digits)}</span></td>`;
        }).join('');
        const bm = ms.find(m => m.key === best);
        const winner = bm ? `<span class="winner-badge winner-${bm.cls}">${esc(bm.name)}</span>` : '<span class="td-note">descriptive only</span>';
        return `<tr><td class="td-label">${esc(mt.label)}${mt.better === 'high' ? ' ↑' : mt.better === 'low' ? ' ↓' : ''}</td>${cells}<td class="td-winner">${winner}</td></tr>`;
    }).join('');

    const floorRow = `<tr><td class="td-label">Runs meeting the ≥20% niche floor</td>${ms.map(m => {
        const runs = data[m.key].per_run_niche_pct;
        const ok = runs.filter(v => v >= 20).length;
        return `<td>${ok} / ${runs.length}</td>`;
    }).join('')}<td class="td-winner"><span class="td-note">floor enforced only by MMR + floor and DPP</span></td></tr>`;

    document.getElementById('comparison-table').innerHTML = head + `<tbody>${rows}${floorRow}</tbody>`;
}

function pairedEntry(data, a, b) {
    const P = data._paired || {};
    if (P[`${a}__minus__${b}`]) return { e: P[`${a}__minus__${b}`], sign: 1 };
    if (P[`${b}__minus__${a}`]) return { e: P[`${b}__minus__${a}`], sign: -1 };
    return null;
}

function renderPaired(data) {
    const pairs = [
        ['graph_dpp_rerank', 'content_filtering'],
        ['mmr_0.7', 'content_filtering'],
        ['graph_dpp_rerank', 'mmr_0.7'],
        ['mmr_floor', 'mmr_0.7'],
        ['mmr_floor', 'graph_dpp_rerank'],
    ];
    const name = k => (METHODS.find(m => m.key === k) || { name: k }).name;
    const cell = (e, sign, metric, digits) => {
        const x = e[metric];
        const sig = x.significant_bonferroni;
        return `<td>${sign * x.mean_diff >= 0 ? '+' : ''}${fmt(sign * x.mean_diff, digits)} / ${sign * x.median_diff >= 0 ? '+' : ''}${fmt(sign * x.median_diff, digits)}` +
               `<span class="td-note">p = ${Number(x.wilcoxon_p).toPrecision(2)} ${sig ? '— survives Bonferroni' : '— not significant after correction'}</span></td>`;
    };
    let alpha = null;
    const body = pairs.map(([a, b]) => {
        const r = pairedEntry(data, a, b);
        if (!r) return '';
        alpha = r.e.ild.bonferroni_alpha;
        return `<tr><td class="td-label">${esc(name(a))} − ${esc(name(b))}</td>${cell(r.e, r.sign, 'ild', 3)}${cell(r.e, r.sign, 'niche_pct', 1)}</tr>`;
    }).join('');
    document.getElementById('paired-table').innerHTML =
        '<thead><tr><th>Comparison (A − B)</th><th>ILD: mean / median diff</th><th>Niche %: mean / median diff</th></tr></thead><tbody>' + body + '</tbody>';
    document.getElementById('paired-note').textContent =
        `Paired two-sided Wilcoxon signed-rank test, same seed sets for every method. Bonferroni threshold across all method pairs and metrics: α = ${alpha}. ` +
        'Raw p-values are shown; only those marked "survives" are claimed as significant.';
}

// Render one method's recommendation list into the search demo
function renderRecList(listEl, recs) {
    listEl.innerHTML = '';
    recs.forEach(rec => {
        const li = document.createElement('li');
        li.className = 'recs-item';
        let popClass = 'pop-med';
        if (rec.popularity >= 80) popClass = 'pop-high';
        else if (rec.popularity < 40) popClass = 'pop-low';
        li.innerHTML = `
            <div class="recs-song">
                <span class="recs-name">${esc(rec.name)}</span>
                <span class="recs-artist">${esc(rec.artist)}</span>
            </div>
            <span class="recs-pop ${popClass}">🔥 ${Math.round(rec.popularity)}</span>
        `;
        listEl.appendChild(li);
    });
}

// Simple particle background animation
function initParticles() {
    const canvas = document.getElementById('particles-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Resize canvas
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    window.addEventListener('resize', resize);
    resize();

    // Create particles
    const particles = [];
    const numParticles = Math.min(50, window.innerWidth / 30);

    for (let i = 0; i < numParticles; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            size: Math.random() * 2 + 0.5,
            speedX: (Math.random() - 0.5) * 0.5,
            speedY: (Math.random() - 0.5) * 0.5,
            opacity: Math.random() * 0.3 + 0.1
        });
    }

    // Animate
    function animate() {
        requestAnimationFrame(animate);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(p => {
            p.x += p.speedX;
            p.y += p.speedY;

            // Wrap around
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(168, 85, 247, ${p.opacity})`;
            ctx.fill();
        });
    }

    animate();
}

// Number counter animation
function animateNumbers() {
    const numbers = document.querySelectorAll('.stat-number');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseInt(el.getAttribute('data-count'));
                let current = 0;

                // Animation duration: 2s
                const duration = 2000;
                const start = performance.now();

                function update(time) {
                    const elapsed = time - start;
                    const progress = Math.min(elapsed / duration, 1);

                    // Easing: easeOutExpo
                    const easeOut = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);

                    current = Math.floor(target * easeOut);

                    // Format with commas if >= 1000
                    el.textContent = current >= 1000 ? current.toLocaleString() : current;

                    if (progress < 1) {
                        requestAnimationFrame(update);
                    } else {
                        el.textContent = target >= 1000 ? target.toLocaleString() : target;
                    }
                }

                requestAnimationFrame(update);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    numbers.forEach(num => observer.observe(num));
}

// Animate metric bars on scroll
function animateBarsOnScroll() {
    const bars = document.querySelectorAll('.metric-bar');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bar = entry.target;
                const targetWidth = bar.style.getPropertyValue('--bar-width');
                // Set brief timeout to allow transition
                setTimeout(() => {
                    bar.style.width = targetWidth;
                }, 100);
                observer.unobserve(bar);
            }
        });
    }, { threshold: 0.1 });

    bars.forEach(bar => {
        bar.style.width = '0%'; // Ensure starting state
        observer.observe(bar);
    });
}

// Now handle charts via Images, since we generated them in Python
// We'll replace the canvas elements with the generated PNGs for stability
document.addEventListener('DOMContentLoaded', () => {

    const chartMapping = {
        'canvas-diversity': 'output/diversity_comparison.png',
        'canvas-fairness': 'output/fairness_comparison.png',
        'canvas-popularity': 'output/popularity_comparison.png',
        'canvas-tradeoff': 'output/tradeoff_chart.png',
        'canvas-niche': 'output/niche_percentage.png',
        'canvas-lambda': 'output/mmr_lambda_curve.png'
    };

    for (const [canvasId, imgSrc] of Object.entries(chartMapping)) {
        const canvas = document.getElementById(canvasId);
        if (canvas) {
            const img = document.createElement('img');
            img.src = imgSrc;
            img.alt = 'Chart';
            img.style.width = '100%';
            img.style.height = 'auto';
            img.style.borderRadius = '8px';
            img.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
            img.style.border = '1px solid rgba(255,255,255,0.05)';
            canvas.parentNode.replaceChild(img, canvas);
        }
    }
});

// Scroll Spy for Nav Links
function initScrollSpy() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');
    const nav = document.getElementById('sticky-nav');

    window.addEventListener('scroll', () => {
        let current = '';

        // Header background transition
        if (window.scrollY > 50) {
            nav.style.background = 'rgba(5, 5, 12, 0.95)';
            nav.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.4)';
        } else {
            nav.style.background = 'rgba(10, 10, 22, 0.8)';
            nav.style.boxShadow = 'none';
        }

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (window.scrollY >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').includes(current)) {
                link.classList.add('active');
            }
        });
    });
}

// Search Feature
function setupSearch() {
    const searchInput = document.getElementById('track-search-input');
    const resultsDropdown = document.getElementById('search-results');
    const clearBtn = document.getElementById('search-clear-btn');
    const infoBox = document.getElementById('demo-seed-info');
    const infoName = document.getElementById('demo-seed-name');

    let debounceTimer;

    if (!searchInput) return;

    // Loading spinner SVG
    const spinnerSvg = `<svg class="search-spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line>
        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
        <line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line>
        <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
    </svg>`;

    // Clear icon SVG
    const clearSvg = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>`;

    function setBtnState(state) {
        if (!clearBtn) return;
        if (state === 'hidden') {
            clearBtn.classList.remove('visible');
            clearBtn.style.pointerEvents = 'none';
        } else if (state === 'clear') {
            clearBtn.classList.add('visible');
            clearBtn.style.pointerEvents = 'auto';
            clearBtn.innerHTML = clearSvg;
        } else if (state === 'loading') {
            clearBtn.classList.add('visible');
            clearBtn.style.pointerEvents = 'none';
            clearBtn.innerHTML = spinnerSvg;
        }
    }

    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        resultsDropdown.classList.remove('active');
        setBtnState('hidden');
        searchInput.focus();
    });

    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        const q = e.target.value.trim();

        if (q.length === 0) {
            resultsDropdown.classList.remove('active');
            setBtnState('hidden');
            return;
        }

        setBtnState('loading');

        debounceTimer = setTimeout(async () => {
            try {
                const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
                if (!res.ok) throw new Error('API down');
                const data = await res.json();

                resultsDropdown.innerHTML = '';
                if (data.length === 0) {
                    resultsDropdown.innerHTML = '<div class="search-result-item">No results found</div>';
                } else {
                    data.forEach(item => {
                        const div = document.createElement('div');
                        div.className = 'search-result-item';
                        div.innerHTML = `
                            <div>
                                <span class="search-result-name">${esc(item.name)}</span>
                                <span class="search-result-artist">${esc(item.artist)}</span>
                            </div>
                            <span class="recs-pop pop-med">🔥 ${Math.round(item.popularity)}</span>
                        `;
                        div.addEventListener('click', () => {
                            searchInput.value = item.name;
                            resultsDropdown.classList.remove('active');
                            setBtnState('clear');
                            fetchRecommendations(item.id, item.name);
                        });
                        resultsDropdown.appendChild(div);
                    });
                }
                resultsDropdown.classList.add('active');
                setBtnState('clear');
            } catch (e) {
                console.error("Search failed:", e);
                resultsDropdown.innerHTML = '<div class="search-result-item" style="color:#ff6b6b">Server error. Check server.py status.</div>';
                resultsDropdown.classList.add('active');
                setBtnState('clear');
            }
        }, 350);
    });

    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !resultsDropdown.contains(e.target)) {
            resultsDropdown.classList.remove('active');
        }
    });

    async function fetchRecommendations(id, name) {
        infoName.textContent = name;
        infoBox.style.display = 'block';
        
        const gridEl = document.getElementById('recs-grid-container');
        if (gridEl) gridEl.style.display = 'grid';

        const algos = METHODS.map(m => m.cls);
        algos.forEach(algo => {
            const listEl = document.getElementById(`recs-list-${algo}`);
            if (listEl) listEl.innerHTML = '<div style="text-align:center; padding: 2rem; color:#888;"><span class="search-spinner" style="display:inline-block; margin-bottom:10px;">' + spinnerSvg + '</span><br>Generating...</div>';
        });

        const demo = document.getElementById('interactive-demo');
        if (demo) demo.scrollIntoView({ behavior: 'smooth', block: 'start' });

        try {
            const res = await fetch(`/api/recommend?id=${id}`);
            const data = await res.json();

            algos.forEach(algo => {
                const listEl = document.getElementById(`recs-list-${algo}`);
                if (!listEl) return;

                renderRecList(listEl, data[algo] || []);
            });
        } catch (e) {
            console.error("Recommendation error:", e);
            alert("Error fetching recommendations. Please make sure the Python server is running.");
        }
    }
}
