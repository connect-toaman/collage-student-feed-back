document.addEventListener('DOMContentLoaded', () => {
    // 0. 3D Tilt Effect - safely initialize if elements exist
    initTiltEffects();

    // 1. Mobile Navigation
    const mobileBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');

    if (mobileBtn && navLinks) {
        mobileBtn.addEventListener('click', () => {
            navLinks.classList.toggle('show');
            const isExpanded = navLinks.classList.contains('show');
            mobileBtn.setAttribute('aria-expanded', isExpanded);
        });
    }

    // 2. Feedback Form Logic (Character count & Validation & Loading state)
    const feedbackForm = document.getElementById('feedbackForm');
    const feedbackText = document.getElementById('feedback_text');
    const charCount = document.getElementById('charCount');
    const submitBtn = document.getElementById('submitBtn');
    
    const MAX_CHARS = 1000;

    if (feedbackText && charCount) {
        feedbackText.addEventListener('input', function() {
            const currentLength = this.value.length;
            charCount.textContent = currentLength;
            
            if (currentLength > MAX_CHARS) {
                charCount.style.color = 'var(--danger)';
                this.value = this.value.substring(0, MAX_CHARS);
                charCount.textContent = MAX_CHARS;
            } else if (currentLength > MAX_CHARS * 0.9) {
                charCount.style.color = 'var(--warning)';
            } else {
                charCount.style.color = 'var(--text-secondary)';
            }
            
            if (currentLength > 0) {
                this.classList.remove('error-field');
                document.getElementById('feedback-error').style.display = 'none';
            }
        });
    }

    if (feedbackForm) {
        feedbackForm.addEventListener('submit', function(e) {
            let isValid = true;
            
            // Basic validation
            const dept = document.getElementById('department');
            const cat = document.getElementById('category');
            const txt = document.getElementById('feedback_text');
            
            if (!dept.value) {
                dept.classList.add('error-field');
                document.getElementById('department-error').style.display = 'block';
                isValid = false;
            } else {
                dept.classList.remove('error-field');
                document.getElementById('department-error').style.display = 'none';
            }
            
            if (!cat.value) {
                cat.classList.add('error-field');
                document.getElementById('category-error').style.display = 'block';
                isValid = false;
            } else {
                cat.classList.remove('error-field');
                document.getElementById('category-error').style.display = 'none';
            }
            
            if (!txt.value.trim()) {
                txt.classList.add('error-field');
                document.getElementById('feedback-error').style.display = 'block';
                isValid = false;
            } else {
                txt.classList.remove('error-field');
                document.getElementById('feedback-error').style.display = 'none';
            }
            
            if (!isValid) {
                e.preventDefault();
                return;
            }

            // Loading state
            if (submitBtn) {
                const btnText = submitBtn.querySelector('.btn-text');
                const loader = submitBtn.querySelector('.loader');
                
                if (btnText && loader) {
                    btnText.textContent = 'Analyzing...';
                    loader.classList.remove('hidden');
                    submitBtn.disabled = true;
                }
            }
        });
        
        // Remove error states on change
        ['department', 'category'].forEach(id => {
            const el = document.getElementById(id);
            if(el) {
                el.addEventListener('change', function() {
                    this.classList.remove('error-field');
                    document.getElementById(`${id}-error`).style.display = 'none';
                });
            }
        });
    }

    // 3. Admin Login Password Toggle
    const togglePassword = document.getElementById('togglePassword');
    const passwordInput = document.getElementById('password');
    
    if (togglePassword && passwordInput) {
        togglePassword.addEventListener('click', function() {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            this.textContent = type === 'password' ? 'Show' : 'Hide';
        });
    }

    // 4. Dashboard Logic (Charts & Filtering)
    const feedbackTable = document.getElementById('feedbackTable');
    if (feedbackTable && typeof Chart !== 'undefined') {
        initDashboard();
    }

    // Initialize loading states
    initLoadingStates();});

function initDashboard() {
    // 1. Setup Filters
    const searchInput = document.getElementById('tableSearch');
    const sentimentFilter = document.getElementById('sentimentFilter');
    const categoryFilter = document.getElementById('categoryFilter');
    const rows = document.querySelectorAll('.feedback-row');
    const noResultsMsg = document.getElementById('noResultsMsg');
    
    // Populate Category Filter dynamically
    if (typeof feedbacks !== 'undefined' && categoryFilter) {
        const uniqueCategories = [...new Set(feedbacks.map(f => f.category))].sort();
        uniqueCategories.forEach(cat => {
            const opt = document.createElement('option');
            opt.value = cat;
            opt.textContent = cat;
            categoryFilter.appendChild(opt);
        });
    }

    function filterTable() {
        const searchTerm = searchInput.value.toLowerCase();
        const sentVal = sentimentFilter.value;
        const catVal = categoryFilter.value;
        
        let visibleCount = 0;

        rows.forEach(row => {
            const text = row.querySelector('.td-text').textContent.toLowerCase();
            const dept = row.querySelector('.td-dept').textContent.toLowerCase();
            const rowSent = row.dataset.sentiment;
            const rowCat = row.dataset.category;
            
            const matchesSearch = text.includes(searchTerm) || dept.includes(searchTerm);
            const matchesSent = sentVal === 'all' || rowSent === sentVal;
            const matchesCat = catVal === 'all' || rowCat === catVal;
            
            if (matchesSearch && matchesSent && matchesCat) {
                row.classList.remove('hidden');
                visibleCount++;
            } else {
                row.classList.add('hidden');
            }
        });
        
        if (visibleCount === 0) {
            document.querySelector('.table-responsive').classList.add('hidden');
            noResultsMsg.classList.remove('hidden');
        } else {
            document.querySelector('.table-responsive').classList.remove('hidden');
            noResultsMsg.classList.add('hidden');
        }
    }

    if (searchInput) searchInput.addEventListener('input', filterTable);
    if (sentimentFilter) sentimentFilter.addEventListener('change', filterTable);
    if (categoryFilter) categoryFilter.addEventListener('change', filterTable);

    // 2. Setup Charts
    if (typeof sentimentData !== 'undefined') {
        // Sentiment Doughnut Chart
        const ctxSent = document.getElementById('sentimentChart').getContext('2d');
        new Chart(ctxSent, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [sentimentData.positive, sentimentData.neutral, sentimentData.negative],
                    backgroundColor: ['#10b981', '#64748b', '#ef4444'],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: { position: 'bottom', labels: { padding: 20, font: { family: "'Inter', sans-serif" } } }
                }
            }
        });

        // Category Bar Chart
        if (typeof feedbacks !== 'undefined') {
            const catCounts = {};
            feedbacks.forEach(f => {
                catCounts[f.category] = (catCounts[f.category] || 0) + 1;
            });
            
            // Sort by count
            const sortedCats = Object.entries(catCounts).sort((a, b) => b[1] - a[1]);
            const labels = sortedCats.map(item => item[0]);
            const data = sortedCats.map(item => item[1]);
            
            const ctxCat = document.getElementById('categoryChart').getContext('2d');
            new Chart(ctxCat, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Feedback Count',
                        data: data,
                        backgroundColor: '#2563eb',
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { beginAtZero: true, ticks: { precision: 0 } },
                        x: { grid: { display: false } }
                    }
                }
            });
        }
    }
}

// 5. Loading & Micro-Interactions
function initLoadingStates() {
    // Initialize loading states
    initLoadingBar();
    initFormLoading();
    initFeedbackSkeleton();
}

function initLoadingBar() {
    const submitBtn = document.getElementById('submitBtn');
    if (submitBtn) {
        submitBtn.style.position = 'relative';
        submitBtn.style.display = 'inline-flex';
        submitBtn.style.width = '100%';
        submitBtn.style.maxWidth = '100%';
        submitBtn.style.minWidth = '100%';
    }
}

function initFormLoading() {
    const feedbackForm = document.getElementById('feedbackForm');
    if (feedbackForm) {
        feedbackForm.addEventListener('submit', function(e) {
            let isValid = true;

            // Basic validation
            const dept = document.getElementById('department');
            const cat = document.getElementById('category');
            const txt = document.getElementById('feedback_text');

            if (!dept.value) {
                dept.classList.add('error-field');
                document.getElementById('department-error').style.display = 'block';
                isValid = false;
            } else {
                dept.classList.remove('error-field');
                document.getElementById('department-error').style.display = 'none';
            }

            if (!cat.value) {
                cat.classList.add('error-field');
                document.getElementById('category-error').style.display = 'block';
                isValid = false;
            } else {
                cat.classList.remove('error-field');
                document.getElementById('category-error').style.display = 'none';
            }

            if (!txt.value.trim()) {
                txt.classList.add('error-field');
                document.getElementById('feedback-error').style.display = 'block';
                isValid = false;
            } else {
                txt.classList.remove('error-field');
                document.getElementById('feedback-error').style.display = 'none';
            }

            if (!isValid) {
                e.preventDefault();
                return;
            }

            // Loading state
            if (submitBtn) {
                const btnText = submitBtn.querySelector('.btn-text');
                const loader = submitBtn.querySelector('.loader');

                if (btnText && loader) {
                    btnText.textContent = 'Analyzing...';
                    loader.classList.remove('hidden');
                    submitBtn.disabled = true;

                    // Add loading bar
                    const loadingBar = document.createElement('div');
                    loadingBar.className = 'loading-bar';
                    submitBtn.parentNode.insertBefore(loadingBar, submitBtn.nextSibling);

                    // Animate loading bar
                    const animateLoading = setInterval(() => {
                        const width = Math.min(100, loadingBar.offsetWidth * 0.95);
                        loadingBar.style.width = `${width}%`;
                        loadingBar.style.transition = 'width 0.5s ease-out';

                        if (width >= 100) {
                            clearInterval(animateLoading);
                            submitBtn.style.transition = 'all 0.3s ease-in-out';
                            submitBtn.style.transform = 'scale(0.98) translateY(-2px)';
                            setTimeout(() => {
                                submitBtn.style.transform = '';
                            }, 300);
                        }
                    }, 100);
                }
            }
        });
    }
}

// 3D Tilt Effect - Subtle hover tilt for cards with interactive-3d class
function initTiltEffects() {
    const cards = document.querySelectorAll('.interactive-3d');
    if (!cards.length) return;

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * 3; // Max 3 degrees
            const rotateY = ((centerX - x) / centerX) * 3; // Max 3 degrees

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });

        card.addEventListener('mousedown', () => {
            card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(0.98, 0.98, 0.98)';
        });

        card.addEventListener('mouseup', () => {
            card.style.transform = '';
        });
    });
}

function initFeedbackSkeleton() {
    const feedbackText = document.getElementById('feedback_text');
    if (feedbackText) {
        feedbackText.style.position = 'relative';
        feedbackText.style.height = 'auto';
        feedbackText.style.padding = '0.75rem 1rem 1.5rem 1rem';
        feedbackText.style.backgroundColor = 'var(--surface)';
        feedbackText.style.border = '1px solid var(--border)';
        feedbackText.style.borderRadius = 'var(--radius-md)';
        feedbackText.style.display = 'inline-block';
        feedbackText.style.width = '100%';
        feedbackText.style.maxHeight = '200px';
        feedbackText.style.overflow = 'hidden';
        feedbackText.style.transition = 'height 0.3s ease-out';
    }
}

function initFormLoading() {
    const feedbackForm = document.getElementById('feedbackForm');
    if (feedbackForm) {
        feedbackForm.addEventListener('submit', function(e) {
            let isValid = true;

            // Basic validation
            const dept = document.getElementById('department');
            const cat = document.getElementById('category');
            const txt = document.getElementById('feedback_text');

            if (!dept.value) {
                dept.classList.add('error-field');
                document.getElementById('department-error').style.display = 'block';
                isValid = false;
            } else {
                dept.classList.remove('error-field');
                document.getElementById('department-error').style.display = 'none';
            }

            if (!cat.value) {
                cat.classList.add('error-field');
                document.getElementById('category-error').style.display = 'block';
                isValid = false;
            } else {
                cat.classList.remove('error-field');
                document.getElementById('category-error').style.display = 'none';
            }

            if (!txt.value.trim()) {
                txt.classList.add('error-field');
                document.getElementById('feedback-error').style.display = 'block';
                isValid = false;
            } else {
                txt.classList.remove('error-field');
                document.getElementById('feedback-error').style.display = 'none';
            }

            if (!isValid) {
                e.preventDefault();
                return;
            }

            // Loading state
            if (submitBtn) {
                const btnText = submitBtn.querySelector('.btn-text');
                const loader = submitBtn.querySelector('.loader');

                if (btnText && loader) {
                    btnText.textContent = 'Analyzing...';
                    loader.classList.remove('hidden');
                    submitBtn.disabled = true;

                    // Add loading bar
                    const loadingBar = document.createElement('div');
                    loadingBar.className = 'loading-bar';
                    submitBtn.parentNode.insertBefore(loadingBar, submitBtn.nextSibling);

                    // Animate loading bar
                    const animateLoading = setInterval(() => {
                        const width = Math.min(100, loadingBar.offsetWidth * 0.95);
                        loadingBar.style.width = `${width}%`;
                        loadingBar.style.transition = 'width 0.5s ease-out';

                        if (width >= 100) {
                            clearInterval(animateLoading);
                            submitBtn.style.transition = 'all 0.3s ease-in-out';
                            submitBtn.style.transform = 'scale(0.98) translateY(-2px)';
                            setTimeout(() => {
                                submitBtn.style.transform = '';
                            }, 300);
                        }
                    }, 100);
                }
            }
        });
    }
}

// 3D Tilt Effect - Subtle hover tilt for cards with interactive-3d class
function initTiltEffects() {
    const cards = document.querySelectorAll('.interactive-3d');
    if (!cards.length) return;

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * 3; // Max 3 degrees
            const rotateY = ((centerX - x) / centerX) * 3; // Max 3 degrees

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });

        card.addEventListener('mousedown', () => {
            card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(0.98, 0.98, 0.98)';
        });

        card.addEventListener('mouseup', () => {
            card.style.transform = '';
        });
    });
}
