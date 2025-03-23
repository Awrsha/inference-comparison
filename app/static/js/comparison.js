// Comparison page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const changeImageBtn = document.getElementById('change-image');
    const runInferenceBtn = document.getElementById('run-inference');
    const resultsSection = document.getElementById('results-section');
    const loadingContainer = document.getElementById('loading-container');
    const resultsGrid = document.getElementById('results-grid');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const timeComparison = document.getElementById('time-comparison');
    const serverTabs = document.querySelectorAll('.server-tab');
    const resultCards = document.querySelectorAll('.result-card');
    const runBenchmarkBtn = document.getElementById('run-benchmark');
    const iterationCount = document.getElementById('iteration-count');
    const benchmarkResults = document.getElementById('benchmark-results');
    
    let selectedFile = null;
    let benchmarkChart = null;
    
    // Event Listeners
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });
    
    changeImageBtn.addEventListener('click', function() {
        resetUI();
    });
    
    runInferenceBtn.addEventListener('click', function() {
        if (selectedFile) {
            runInference();
        }
    });
    
    serverTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const server = this.getAttribute('data-server');
            
            // Update active tab
            serverTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Show/hide cards based on selected server
            resultCards.forEach(card => {
                if (server === 'all' || card.getAttribute('data-server') === server) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
    
    runBenchmarkBtn.addEventListener('click', function() {
        runBenchmark();
    });
    
    // Functions
    function handleFile(file) {
        // Check if file is an image
        if (!file.type.match('image.*')) {
            showError('لطفاً یک فایل تصویری انتخاب کنید (jpg، png)');
            return;
        }
        
        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewContainer.style.display = 'block';
            
            // Reset results if they were shown
            resultsGrid.style.display = 'none';
            timeComparison.style.display = 'none';
            errorMessage.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
    
    function resetUI() {
        uploadArea.style.display = 'block';
        previewContainer.style.display = 'none';
        resultsSection.style.display = 'none';
        selectedFile = null;
        fileInput.value = '';
    }
    
    function runInference() {
        // Show results section with loading state
        resultsSection.style.display = 'block';
        loadingContainer.style.display = 'flex';
        resultsGrid.style.display = 'none';
        timeComparison.style.display = 'none';
        errorMessage.style.display = 'none';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Make API request
        fetch('/api/inference/all', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            loadingContainer.style.display = 'none';
            
            // Check if any server succeeded
            const anySuccess = Object.values(data).some(result => result.success);
            
            if (anySuccess) {
                displayResults(data);
            } else {
                // All servers failed
                const errors = Object.entries(data)
                    .map(([server, result]) => `${server}: ${result.error}`)
                    .join('<br>');
                
                showError(`همه سرورها با خطا مواجه شدند:<br>${errors}`);
            }
        })
        .catch(error => {
            loadingContainer.style.display = 'none';
            showError('خطا در ارتباط با سرور: ' + error.message);
        });
    }
    
    function displayResults(data) {
        // Show results grid
        resultsGrid.style.display = 'grid';
        
        // Process and display each server's results
        for (const [server, result] of Object.entries(data)) {
            const predictionsContainer = document.getElementById(`${server}-predictions`);
            const timeElement = document.getElementById(`${server}-time`);
            
            // Clear previous results
            predictionsContainer.innerHTML = '';
            
            if (result.success) {
                // Display time
                timeElement.textContent = `${(result.inference_time * 1000).toFixed(1)} ms`;
                
                // Display predictions
                result.result.forEach(prediction => {
                    const predictionElement = document.createElement('div');
                    predictionElement.className = 'prediction-item';
                    
                    const probability = (prediction.probability * 100).toFixed(1);
                    
                    predictionElement.innerHTML = `
                        <div class="prediction-label">${prediction.label}</div>
                        <div class="prediction-probability">
                            <div class="probability-bar" style="width: ${probability}px;"></div>
                            <span>${probability}%</span>
                        </div>
                    `;
                    
                    predictionsContainer.appendChild(predictionElement);
                });
            } else {
                // Display error in this server's card
                predictionsContainer.innerHTML = `
                    <div class="server-error">
                        <i class="fas fa-exclamation-circle"></i>
                        <p>${result.error}</p>
                    </div>
                `;
            }
        }
        
        // Display time comparison
        updateTimeComparison(data);
    }
    
    function updateTimeComparison(data) {
        // Get inference times
        const times = {
            triton: data.triton.success ? data.triton.inference_time * 1000 : 0,
            pytorch: data.pytorch.success ? data.pytorch.inference_time * 1000 : 0,
            torchserve: data.torchserve.success ? data.torchserve.inference_time * 1000 : 0
        };
        
        // If all failed, don't show comparison
        if (times.triton === 0 && times.pytorch === 0 && times.torchserve === 0) {
            return;
        }
        
        // Find the maximum time for scaling
        const maxTime = Math.max(...Object.values(times).filter(t => t > 0));
        
        // Update bars
        for (const [server, time] of Object.entries(times)) {
            const bar = document.getElementById(`${server}-bar`);
            const barValue = document.getElementById(`${server}-bar-value`);
            
            if (time > 0) {
                const percentage = (time / maxTime) * 100;
                bar.style.width = `${percentage}%`;
                barValue.textContent = `${time.toFixed(1)} ms`;
            } else {
                bar.style.width = '0%';
                barValue.textContent = 'خطا';
            }
        }
        
        timeComparison.style.display = 'block';
    }
    
    function showError(message) {
        errorText.innerHTML = message;
        errorMessage.style.display = 'flex';
        resultsGrid.style.display = 'none';
        timeComparison.style.display = 'none';
    }
    
    function runBenchmark() {
        // Validate iteration count
        const iterations = parseInt(iterationCount.value);
        if (isNaN(iterations) || iterations < 1 || iterations > 100) {
            alert('لطفاً تعداد تکرار معتبر بین 1 تا 100 وارد کنید');
            return;
        }
        
        // Change button state
        runBenchmarkBtn.disabled = true;
        runBenchmarkBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال اجرا...';
        
        // Clear previous results
        benchmarkResults.style.display = 'none';
        
        // Make API request
        fetch('/api/benchmark', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                iterations: iterations
            })
        })
        .then(response => response.json())
        .then(data => {
            displayBenchmarkResults(data, iterations);
        })
        .catch(error => {
            alert('خطا در انجام آزمایش: ' + error.message);
        })
        .finally(() => {
            // Reset button
            runBenchmarkBtn.disabled = false;
            runBenchmarkBtn.innerHTML = 'شروع آزمایش';
        });
    }
    
    function displayBenchmarkResults(data, iterations) {
        benchmarkResults.style.display = 'block';
        
        // Create summary
        const summaryContainer = document.getElementById('benchmark-summary');
        summaryContainer.innerHTML = '';
        
        for (const [server, results] of Object.entries(data)) {
            const avgTime = results.avg * 1000; // Convert to ms
            const successfulIterations = results.times.length;
            const errorRate = (results.errors / iterations) * 100;
            
            const summaryCard = document.createElement('div');
            summaryCard.className = 'summary-card';
            summaryCard.innerHTML = `
                <h4>${capitalizeFirstLetter(server)}</h4>
                <div class="summary-value">${avgTime.toFixed(1)} ms</div>
                <div class="summary-label">میانگین زمان</div>
                <hr style="margin: 10px 0; border-color: var(--border);">
                <div class="summary-detail">
                    <div>موفق: ${successfulIterations}/${iterations}</div>
                    <div>نرخ خطا: ${errorRate.toFixed(1)}%</div>
                </div>
            `;
            
            summaryContainer.appendChild(summaryCard);
        }
        
        // Create chart
        createBenchmarkChart(data);
    }
    
    function createBenchmarkChart(data) {
        const ctx = document.getElementById('benchmark-chart').getContext('2d');
        
        // Destroy previous chart if exists
        if (benchmarkChart) {
            benchmarkChart.destroy();
        }
        
        // Prepare data
        const labels = [];
        const dataSets = [];
        
        // Create labels (iteration numbers)
        const maxIterations = Math.max(...Object.values(data).map(d => d.times.length));
        for (let i = 1; i <= maxIterations; i++) {
            labels.push(`#${i}`);
        }
        
        // Colors for each server
        const colors = {
            triton: 'rgba(93, 92, 222, 0.7)',
            pytorch: 'rgba(255, 99, 132, 0.7)',
            torchserve: 'rgba(75, 192, 192, 0.7)'
        };
        
        // Create datasets
        for (const [server, results] of Object.entries(data)) {
            if (results.times.length > 0) {
                dataSets.push({
                    label: capitalizeFirstLetter(server),
                    data: results.times.map(t => t * 1000), // Convert to ms
                    backgroundColor: colors[server],
                    borderColor: colors[server].replace('0.7', '1'),
                    borderWidth: 2,
                    tension: 0.2
                });
            }
        }
        
        // Create chart
        benchmarkChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: dataSets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: 'زمان استنباط (میلی‌ثانیه)'
                        },
                        beginAtZero: true
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'شماره تکرار'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)} ms`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});