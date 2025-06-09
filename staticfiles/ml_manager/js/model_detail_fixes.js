/* JavaScript fixes for model_detail.html template */

// Fixed log view switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const logViewButtons = document.querySelectorAll('input[name="logView"]');
    const logViews = document.querySelectorAll('.log-view');
    let liveLogInterval = null;
    
    // Log view switching
    logViewButtons.forEach(function(button) {
        button.addEventListener('change', function() {
            if (this.checked) {
                // Hide all log views
                logViews.forEach(function(view) {
                    view.style.display = 'none';
                    view.classList.remove('active');
                });
                
                // Show selected view
                let viewId = this.id.replace('logView', '').toLowerCase() + '-logs';
                
                // Special handling for live logs
                if (this.id === 'logViewLive') {
                    viewId = 'live-logs';
                }
                
                const targetView = document.getElementById(viewId);
                if (targetView) {
                    targetView.style.display = 'block';
                    targetView.classList.add('active');
                }
                
                const currentLogView = this.id.replace('logView', '').toLowerCase();
                
                // Show/hide model config section for 'all' view
                const configSection = document.getElementById('model-config-section');
                if (currentLogView === 'all' && configSection) {
                    configSection.style.display = 'block';
                } else if (configSection) {
                    configSection.style.display = 'none';
                }
                
                // Handle live log fetching
                if (currentLogView === 'live') {
                    startLiveLogFetching();
                } else {
                    stopLiveLogFetching();
                }
            }
        });
    });
    
    // Live log fetching functions with enhanced error handling
    let logFetchRetries = 0;
    const maxRetries = 3;
    let lastLogLines = [];
    
    function startLiveLogFetching() {
        if (liveLogInterval) {
            clearInterval(liveLogInterval);
        }
        
        logFetchRetries = 0;
        fetchTrainingLog(); // Initial fetch
        liveLogInterval = setInterval(fetchTrainingLog, 2000);
    }
    
    function stopLiveLogFetching() {
        if (liveLogInterval) {
            clearInterval(liveLogInterval);
            liveLogInterval = null;
        }
        logFetchRetries = 0;
    }
    
    function fetchTrainingLog() {
        const logDiv = document.getElementById('training-log');
        if (!logDiv) return;
        
        // Get model ID from template
        const modelId = window.modelId || document.querySelector('[data-model-id]')?.dataset.modelId;
        if (!modelId) {
            console.warn('Model ID not found for log fetching');
            return;
        }
        
        // Show loading indicator for first fetch or after errors
        if (logFetchRetries === 0 || lastLogLines.length === 0) {
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'text-center text-muted py-2';
            loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading logs...';
            logDiv.appendChild(loadingDiv);
        }
        
        fetch(window.location.href, {
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
            .then(function(response) {
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status);
                }
                return response.json();
            })
            .then(function(data) {
                // Reset retry counter on success
                logFetchRetries = 0;
                
                // Update logs
                if (data.logs && Array.isArray(data.logs)) {
                    // Clear loading indicator
                    logDiv.innerHTML = '';
                    
                    // Only update if logs have changed
                    if (JSON.stringify(data.logs) !== JSON.stringify(lastLogLines)) {
                        lastLogLines = data.logs;
                        updateLogDisplay(data);
                    }
                    
                    // Auto-scroll to bottom for new content
                    logDiv.scrollTop = logDiv.scrollHeight;
                } else {
                    displayLogError('No log data available');
                }
                
                // Update progress if available
                if (data.progress) {
                    updateProgress(data);
                }
            })
            .catch(function(error) {
                console.error('Log fetch error:', error);
                logFetchRetries++;
                
                if (logFetchRetries <= maxRetries) {
                    // Show retry message
                    displayLogError('Connection error, retrying... (' + logFetchRetries + '/' + maxRetries + ')');
                    
                    // Exponential backoff
                    setTimeout(fetchTrainingLog, 1000 * logFetchRetries);
                } else {
                    displayLogError('Failed to load logs after ' + maxRetries + ' attempts. Check connection.');
                    stopLiveLogFetching();
                }
            });
    }
    
    function displayLogError(message) {
        const logDiv = document.getElementById('training-log');
        if (logDiv) {
            logDiv.innerHTML = '<div class="text-center text-warning py-3"><i class="fas fa-exclamation-triangle"></i> ' + message + '</div>';
        }
    }
    
    function updateLogDisplay(data) {
        // Update different log views
        const logViews = ['all', 'epoch', 'batch', 'metrics', 'live'];
        
        logViews.forEach(function(viewType) {
            const viewDiv = document.getElementById(viewType + '-logs');
            if (!viewDiv) return;
            
            viewDiv.innerHTML = '';
            
            let logsToShow = [];
            if (viewType === 'all' || viewType === 'live') {
                logsToShow = data.logs || [];
            } else if (data.parsed_logs && data.parsed_logs[viewType + '_logs']) {
                logsToShow = data.parsed_logs[viewType + '_logs'];
            }
            
            if (logsToShow.length > 0) {
                logsToShow.forEach(function(line) {
                    const logEntry = document.createElement('div');
                    logEntry.className = 'log-entry';
                    
                    // Add appropriate styling based on log type
                    if (line.includes('[EPOCH]')) {
                        logEntry.classList.add('epoch-log');
                    } else if (line.includes('[TRAIN]') || line.includes('[VAL]')) {
                        logEntry.classList.add(line.includes('[VAL]') ? 'validation-log' : 'batch-log');
                    } else if (line.includes('[METRICS]')) {
                        logEntry.classList.add('metrics-log');
                    } else if (line.includes('[CONFIG]')) {
                        logEntry.classList.add('config-log');
                    } else if (line.includes('[MODEL]')) {
                        logEntry.classList.add('model-log');
                    }
                    
                    logEntry.textContent = line;
                    viewDiv.appendChild(logEntry);
                });
            } else {
                const emptyMsg = document.createElement('div');
                emptyMsg.className = 'text-center text-muted py-3';
                emptyMsg.innerHTML = '<i class="fas fa-info-circle"></i> No ' + viewType + ' logs available yet.';
                viewDiv.appendChild(emptyMsg);
            }
        });
        
        // Update log statistics
        if (data.log_stats) {
            updateLogStats(data.log_stats);
        }
    }
    
    function updateLogStats(stats) {
        const statsElements = {
            'total-lines': stats.total_count,
            'epoch-count': stats.epoch_count,
            'batch-count': stats.batch_count,
            'metrics-count': stats.metrics_count,
            'last-update': stats.last_update
        };
        
        Object.keys(statsElements).forEach(function(id) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = statsElements[id];
            }
        });
    }
    
    // Initialize with default view if training model
    const isTraining = document.querySelector('[data-model-status="training"]');
    if (isTraining) {
        // Set up live log fetching for training models
        const liveViewButton = document.getElementById('logViewLive');
        if (liveViewButton) {
            // Switch to live view for training models
            liveViewButton.checked = true;
            liveViewButton.dispatchEvent(new Event('change'));
        }
    }
});

// Enhanced progress update functionality
function updateProgress(data) {
    const progress = data.progress;
    if (!progress) return;
    
    // Update progress bar
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        const percentage = progress.total_epochs > 0 ? 
            (progress.current_epoch / progress.total_epochs) * 100 : 0;
        progressBar.style.width = percentage + '%';
        progressBar.textContent = `Epoch ${progress.current_epoch}/${progress.total_epochs} (${percentage.toFixed(1)}%)`;
        progressBar.setAttribute('aria-valuenow', progress.current_epoch);
    }
    
    // Update individual metric elements
    const metricUpdates = {
        'current-epoch': progress.current_epoch,
        'total-epochs': progress.total_epochs,
        'train-loss': progress.train_loss ? progress.train_loss.toFixed(4) : 'N/A',
        'val-loss': progress.val_loss ? progress.val_loss.toFixed(4) : 'N/A',
        'train-dice': progress.train_dice ? progress.train_dice.toFixed(4) : 'N/A',
        'val-dice': progress.val_dice ? progress.val_dice.toFixed(4) : 'N/A',
        'best-val-dice': progress.best_val_dice ? progress.best_val_dice.toFixed(4) : 'N/A'
    };
    
    Object.keys(metricUpdates).forEach(function(id) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = metricUpdates[id];
            
            // Add visual feedback for important metrics
            if (id === 'best-val-dice' && progress.best_val_dice) {
                element.parentElement.classList.add('text-success');
                element.classList.add('fw-bold');
            }
        }
    });
    
    // Update status if model completed
    if (data.status && data.status !== 'training') {
        const statusBadge = document.querySelector('.badge');
        if (statusBadge) {
            statusBadge.textContent = data.status;
            statusBadge.className = `badge ${data.status === 'completed' ? 'bg-success' : 'bg-danger'}`;
        }
        
        // Stop live updates if no longer training
        if (data.status !== 'training') {
            stopLiveLogFetching();
            clearInterval(window.progressInterval);
        }
    }
}

// Enhanced auto refresh for training models
function refreshProgress() {
    const isTraining = document.querySelector('[data-model-status="training"]');
    if (!isTraining) return;
    
    const modelId = window.modelId || document.querySelector('[data-model-id]')?.dataset.modelId;
    if (!modelId) return;
    
    fetch(window.location.href, {
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
        .then(function(response) {
            if (!response.ok) {
                throw new Error('HTTP ' + response.status);
            }
            return response.json();
        })
        .then(function(data) {
            updateProgress(data);
            
            // Update logs if live view is active
            const liveViewActive = document.getElementById('logViewLive')?.checked;
            if (liveViewActive && data.logs) {
                updateLogDisplay(data);
            }
        })
        .catch(function(error) {
            console.error('Error fetching progress:', error);
            
            // Show error in UI
            const progressContainer = document.getElementById('training-progress');
            if (progressContainer) {
                const errorDiv = progressContainer.querySelector('.connection-error') || 
                               document.createElement('div');
                errorDiv.className = 'alert alert-warning connection-error mt-2';
                errorDiv.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Connection error updating progress';
                
                if (!progressContainer.querySelector('.connection-error')) {
                    progressContainer.appendChild(errorDiv);
                }
                
                // Remove error message after 5 seconds
                setTimeout(() => {
                    if (errorDiv.parentNode) {
                        errorDiv.parentNode.removeChild(errorDiv);
                    }
                }, 5000);
            }
        });
}

// Start auto refresh for training models with better management
document.addEventListener('DOMContentLoaded', function() {
    const isTraining = document.querySelector('[data-model-status="training"]');
    if (isTraining) {
        // Store interval ID globally for cleanup
        window.progressInterval = setInterval(refreshProgress, 3000);
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            if (window.progressInterval) {
                clearInterval(window.progressInterval);
            }
            stopLiveLogFetching();
        });
    }
    
    // Add model ID to window for easier access
    const modelElement = document.querySelector('[data-model-id]');
    if (modelElement) {
        window.modelId = modelElement.dataset.modelId;
    }
});
