/* JavaScript fixes for model_detail.html template with manual log viewing */

// Fixed log view switching functionality with manual log viewing
document.addEventListener('DOMContentLoaded', function() {
    const logViewButtons = document.querySelectorAll('input[name="logView"]');
    const logViews = document.querySelectorAll('.log-view');
    
    // Initialize ModelLogsManager for AJAX-based log filtering
    let logManager = null;
    const modelId = document.querySelector('[data-model-id]')?.getAttribute('data-model-id');
    
    if (modelId && typeof ModelLogsManager !== 'undefined') {
        try {
            logManager = new ModelLogsManager(modelId);
            console.log('ModelLogsManager initialized for AJAX log filtering');
        } catch (error) {
            console.warn('ModelLogsManager not available:', error);
        }
    }
    
    // Add event listener for View Logs button
    const viewLogsButton = document.getElementById('viewLogsBtn');
    if (viewLogsButton) {
        viewLogsButton.addEventListener('click', function() {
            fetchCurrentLogs();
        });
    }
    
    // Log view switching without live polling
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
                
                // If using AJAX filtering, update via ModelLogsManager
                if (logManager) {
                    console.log('AJAX log filtering triggered for view:', currentLogView);
                }
            }
        });
    });
    
    // Manual log fetching function
    function fetchCurrentLogs() {
        // Try to find the appropriate log container based on current view
        let logDiv = null;
        
        // First, check which log view is currently active
        const activeLogView = document.querySelector('input[name="logView"]:checked');
        if (activeLogView) {
            const viewType = activeLogView.id.replace('logView', '').toLowerCase();
            logDiv = document.getElementById(viewType + '-logs');
        }
        
        // Fallback to common container IDs if no active view found
        if (!logDiv) {
            const containerIds = ['live-logs', 'all-logs', 'training-log', 'training-logs'];
            for (const id of containerIds) {
                logDiv = document.getElementById(id);
                if (logDiv) break;
            }
        }
        
        // Final fallback to any element with log-related classes
        if (!logDiv) {
            logDiv = document.querySelector('.log-display, .log-section');
        }
        
        if (!logDiv) {
            console.error('No log container found');
            return;
        }
        
        // Get model ID from template
        const modelId = window.modelId || document.querySelector('[data-model-id]')?.dataset.modelId;
        if (!modelId) {
            console.warn('Model ID not found for log fetching');
            return;
        }
        
        // Show loading indicator
        logDiv.innerHTML = '<div class="text-center text-muted py-2"><i class="fas fa-spinner fa-spin"></i> Loading logs...</div>';
        
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
                console.log('AJAX response received:', data);
                
                // Update logs
                if (data.logs && Array.isArray(data.logs)) {
                    updateLogDisplay(data);
                    
                    // Auto-scroll to bottom for new content
                    logDiv.scrollTop = logDiv.scrollHeight;
                } else {
                    logDiv.innerHTML = '<div class="text-center text-warning py-3"><i class="fas fa-info-circle"></i> No log data available</div>';
                }
                
                // Update progress if available
                if (data.progress) {
                    updateProgress(data);
                }
            })
            .catch(function(error) {
                console.error('Log fetch error:', error);
                logDiv.innerHTML = '<div class="text-center text-warning py-3"><i class="fas fa-exclamation-triangle"></i> Failed to load logs. Please try again.</div>';
            });
    }
    
    function updateLogDisplay(data) {
        console.log('Updating log display with data:', data);
        
        // Update different log views
        const logViews = ['all', 'epoch', 'batch', 'metrics'];
        
        logViews.forEach(function(viewType) {
            const viewDiv = document.getElementById(viewType + '-logs');
            if (!viewDiv) return;
            
            // Clear the container
            viewDiv.innerHTML = '';
            
            let logsToShow = [];
            if (viewType === 'all') {
                logsToShow = data.logs || [];
            } else if (data.parsed_logs && data.parsed_logs[viewType + '_logs']) {
                logsToShow = data.parsed_logs[viewType + '_logs'];
            }
            
            if (logsToShow.length > 0) {
                // Create a container for the logs
                const logContainer = document.createElement('div');
                logContainer.className = 'log-section';
                
                logsToShow.forEach(function(logItem) {
                    const logEntry = document.createElement('div');
                    logEntry.className = 'log-entry';
                    
                    // Handle both string and object formats
                    let lineContent = '';
                    let logLevel = 'INFO';
                    
                    if (typeof logItem === 'string') {
                        lineContent = logItem;
                    } else if (typeof logItem === 'object' && logItem !== null) {
                        lineContent = logItem.content || logItem.toString();
                        logLevel = logItem.level || 'INFO';
                    } else {
                        lineContent = String(logItem);
                    }
                    
                    // Add appropriate styling based on log type
                    if (lineContent.includes('[EPOCH]') || logLevel === 'EPOCH') {
                        logEntry.classList.add('epoch-log');
                    } else if (lineContent.includes('[TRAIN]') || lineContent.includes('[VAL]') || logLevel === 'TRAINING') {
                        logEntry.classList.add(lineContent.includes('[VAL]') ? 'validation-log' : 'batch-log');
                    } else if (lineContent.includes('[METRICS]') || logLevel === 'METRICS') {
                        logEntry.classList.add('metrics-log');
                    } else if (lineContent.includes('[CONFIG]')) {
                        logEntry.classList.add('config-log');
                    } else if (lineContent.includes('[MODEL]')) {
                        logEntry.classList.add('model-log');
                    } else if (lineContent.includes('ERROR') || logLevel === 'ERROR') {
                        logEntry.classList.add('error-log');
                    } else if (lineContent.includes('WARNING') || logLevel === 'WARNING') {
                        logEntry.classList.add('warning-log');
                    }
                    
                    logEntry.textContent = lineContent;
                    logContainer.appendChild(logEntry);
                });
                
                viewDiv.appendChild(logContainer);
            } else {
                const emptyMsg = document.createElement('div');
                emptyMsg.className = 'text-center text-muted py-3';
                emptyMsg.innerHTML = '<i class="fas fa-info-circle"></i> No ' + viewType + ' logs available yet.';
                viewDiv.appendChild(emptyMsg);
            }
        });
        
        // Update live logs container specifically (for training models)
        const liveLogDiv = document.getElementById('live-logs');
        if (liveLogDiv && data.logs) {
            // Find the inner training-log container
            let innerLogDiv = liveLogDiv.querySelector('#training-log');
            if (!innerLogDiv) {
                // Create it if it doesn't exist
                innerLogDiv = document.createElement('div');
                innerLogDiv.id = 'training-log';
                innerLogDiv.className = 'log-section log-display';
                innerLogDiv.style.cssText = 'min-height: 180px; max-height: 400px; overflow-y: auto;';
                liveLogDiv.appendChild(innerLogDiv);
            }
            
            innerLogDiv.innerHTML = '';
            
            if (data.logs.length > 0) {
                data.logs.forEach(function(logItem) {
                    const logEntry = document.createElement('div');
                    logEntry.className = 'log-entry';
                    
                    let lineContent = '';
                    if (typeof logItem === 'string') {
                        lineContent = logItem;
                    } else if (typeof logItem === 'object' && logItem !== null) {
                        lineContent = logItem.content || logItem.toString();
                    } else {
                        lineContent = String(logItem);
                    }
                    
                    logEntry.textContent = lineContent;
                    innerLogDiv.appendChild(logEntry);
                });
                
                // Auto-scroll to bottom
                innerLogDiv.scrollTop = innerLogDiv.scrollHeight;
            } else {
                innerLogDiv.innerHTML = '<div class="text-center text-muted py-3"><i class="fas fa-info-circle"></i> No logs available yet.</div>';
            }
        }
        
        // Update the main training-logs container (for pre-formatted display)
        const mainTrainingLog = document.getElementById('training-logs');
        if (mainTrainingLog && data.logs) {
            const logText = data.logs.map(logItem => {
                if (typeof logItem === 'string') {
                    return logItem;
                } else if (typeof logItem === 'object' && logItem !== null) {
                    return logItem.content || logItem.toString();
                } else {
                    return String(logItem);
                }
            }).join('\n');
            
            mainTrainingLog.textContent = logText;
        }
        
        // Update log statistics
        if (data.log_stats) {
            updateLogStats(data.log_stats);
        }
        
        console.log('Log display update completed');
    }
    
    function updateLogStats(stats) {
        const statsElements = {
            'total-lines': stats.total_count || stats.total_lines,
            'epoch-count': stats.epoch_count || stats.epoch_logs,
            'batch-count': stats.batch_count || stats.batch_logs,
            'metrics-count': stats.metrics_count || stats.model_logs,
            'last-update': stats.last_update || stats.last_updated
        };
        
        Object.keys(statsElements).forEach(function(id) {
            const element = document.getElementById(id);
            if (element && statsElements[id] !== undefined) {
                element.textContent = statsElements[id];
            }
        });
        
        // Update filter info
        const filterInfo = document.getElementById('filter-info');
        if (filterInfo) {
            const activeLogView = document.querySelector('input[name="logView"]:checked');
            const viewType = activeLogView ? activeLogView.id.replace('logView', '').toLowerCase() : 'all';
            filterInfo.textContent = `Type: ${viewType}`;
        }
    }
    
    // Initialize with default view (no automatic live log fetching)
    const defaultView = document.getElementById('logViewAll');
    if (defaultView) {
        defaultView.checked = true;
        defaultView.dispatchEvent(new Event('change'));
    }
});

// Basic progress update functionality (no automatic polling)
function updateProgress(data) {
    // Add comprehensive error checking
    if (!data) {
        console.warn('updateProgress called with no data');
        return;
    }
    
    const progress = data.progress;
    if (!progress) {
        console.warn('updateProgress called with no progress data');
        return;
    }
    
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
    }
}

// Setup model ID for easier access
document.addEventListener('DOMContentLoaded', function() {
    // Add model ID to window for easier access
    const modelElement = document.querySelector('[data-model-id]');
    if (modelElement) {
        window.modelId = modelElement.dataset.modelId;
    }
});
