// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    animateNumbers();
    loadResultsData();
    animateBarsOnScroll();
    initScrollSpy();
    setupSearch();
});

// Load the JSON results and display sample recommendations
async function loadResultsData() {
    try {
        // Fetch the output JSON. In a production app, this would be an API call.
        // For static deployment, we expect it to be served alongside the HTML.
        const response = await fetch('output/results.json');

        if (!response.ok) {
            console.error('Failed to load results.json, using fallback mock data for testing UI.');
            useFallbackData();
            return;
        }

        const data = await response.json();
        renderRecommendations(data);
    } catch (e) {
        console.error('Error fetching results.json:', e);
        // Fallback for local file testing without server
        useFallbackData();
    }
}

// Fallback data if JSON cannot be loaded (e.g. file:// protocol testing)
function useFallbackData() {
    const fallback = {
        "greedy": {
            "runs": [{
                "recommendations": [
                    { "name": "Bright Dark Roses", "artist": "21 Savage", "popularity": 98 },
                    { "name": "Stars of Stars", "artist": "Kendrick Lamar", "popularity": 97 },
                    { "name": "The Memories", "artist": "Kanye West", "popularity": 97 },
                    { "name": "Broken Crystal Fire", "artist": "Kendrick Lamar", "popularity": 97 },
                    { "name": "Hidden Heart", "artist": "21 Savage", "popularity": 96 }
                ]
            }]
        },
        "content_filtering": {
            "runs": [{
                "recommendations": [
                    { "name": "Shadows of Roses", "artist": "Daniel Caesar", "popularity": 41 },
                    { "name": "The Skies", "artist": "Jason Aldean", "popularity": 46 },
                    { "name": "Sacred", "artist": "Beach House", "popularity": 31 },
                    { "name": "Endless Eyes", "artist": "Luke Bryan", "popularity": 38 },
                    { "name": "The Flames", "artist": "Brent Faiyaz", "popularity": 35 }
                ]
            }]
        },
        "graph_dpp_rerank": {
            "runs": [{
                "recommendations": [
                    { "name": "Shadows of Roses", "artist": "Daniel Caesar", "popularity": 41 },
                    { "name": "The Skies", "artist": "Jason Aldean", "popularity": 46 },
                    { "name": "Endless Eyes", "artist": "Luke Bryan", "popularity": 38 },
                    { "name": "Sacred", "artist": "Beach House", "popularity": 31 },
                    { "name": "Bright Midnight Flames", "artist": "Tame Impala", "popularity": 28 }
                ]
            }]
        }
    };
    renderRecommendations(fallback);
}

// Render the sample recommendations lists
function renderRecommendations(data) {
    const algos = ['greedy', 'content_filtering', 'graph_dpp_rerank'];

    algos.forEach(algo => {
        const listEl = document.getElementById(`recs-list-${algo}`);
        if (!listEl) return;

        listEl.innerHTML = ''; // Clear default

        // Take first run
        const run = data[algo].runs[0];

        // Take top 5 for UI clarity
        run.recommendations.slice(0, 5).forEach(rec => {
            const li = document.createElement('li');
            li.className = 'recs-item';

            let popClass = 'pop-med';
            if (rec.popularity >= 80) popClass = 'pop-high';
            else if (rec.popularity < 40) popClass = 'pop-low';

            li.innerHTML = `
                <div class="recs-song">
                    <span class="recs-name">${rec.name}</span>
                    <span class="recs-artist">${rec.artist}</span>
                </div>
                <span class="recs-pop ${popClass}">🔥 ${rec.popularity}</span>
            `;

            listEl.appendChild(li);
        });
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
        'canvas-niche': 'output/niche_percentage.png'
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
                                <span class="search-result-name">${item.name}</span>
                                <span class="search-result-artist">${item.artist}</span>
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

        const algos = ['greedy', 'content_filtering', 'graph_dpp_rerank'];
        algos.forEach(algo => {
            const listEl = document.getElementById(`recs-list-${algo}`);
            if (listEl) listEl.innerHTML = '<div style="text-align:center; padding: 2rem; color:#888;"><span class="search-spinner" style="display:inline-block; margin-bottom:10px;">' + spinnerSvg + '</span><br>Generating...</div>';
        });

        document.getElementById('interactive-demo').scrollIntoView({ behavior: 'smooth', block: 'start' });

        try {
            const res = await fetch(`/api/recommend?id=${id}`);
            const data = await res.json();

            algos.forEach(algo => {
                const listEl = document.getElementById(`recs-list-${algo}`);
                if (!listEl) return;

                listEl.innerHTML = '';
                const recs = data[algo] || [];

                recs.forEach(rec => {
                    const li = document.createElement('li');
                    li.className = 'recs-item';

                    let popClass = 'pop-med';
                    if (rec.popularity >= 80) popClass = 'pop-high';
                    else if (rec.popularity < 40) popClass = 'pop-low';

                    li.innerHTML = `
                        <div class="recs-song">
                            <span class="recs-name">${rec.name}</span>
                            <span class="recs-artist">${rec.artist}</span>
                        </div>
                        <span class="recs-pop ${popClass}">🔥 ${Math.round(rec.popularity)}</span>
                    `;

                    listEl.appendChild(li);
                });
            });
        } catch (e) {
            console.error("Recommendation error:", e);
            alert("Error fetching recommendations. Please make sure the Python server is running.");
        }
    }
}
