/**
 * Simplified Model Detail UI Manager
 * Fixes flickering indicators and adds epoch navigation
 */

class ModelDetailManager {
    constructor(modelId) {
        this.modelId = modelId;
        this.updateInterval = null;
        this.isTraining = false;
        this.updateIntervalMs = 2000;
        this.lastKnownStatus = null;
        this.statusTransitionDetected = false;
        
        this.elements = {
            updateStatus: document.getElementById('update-status'),
            lastUpdateTime: document.getElementById('last-update-time'),
            progressBar: document.querySelector('.progress-bar'),
            batchProgressBar: document.querySelector('.progress-bar.bg-info'),
            stopBtn: document.getElementById('stopTrainingBtn'),
            manualRefreshBtn: document.getElementById('manualRefreshBtn')
        };
        
        // Create debug panel
        this.createDebugPanel();
        
        this.init();
    }
    
    createDebugPanel() {
        // Create a debug panel that shows at the top of the page
        const debugPanel = document.createElement('div');
        debugPanel.id = 'debug-panel';
        debugPanel.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 12px;
            z-index: 9999;
            max-width: 400px;
            max-height: 300px;
            overflow-y: auto;
            display: none;
        `;
        debugPanel.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong>🔍 Enhanced Auto-Refresh Debug</strong>
                <button id="close-debug-panel" style="background: none; border: none; color: white; cursor: pointer; font-size: 16px;">&times;</button>
            </div>
            <div style="font-size: 10px; color: #ccc; margin-bottom: 5px;">Status Transition Detection: ACTIVE</div>
            <div id="debug-log"></div>
        `;
        document.body.appendChild(debugPanel);
        
        this.debugLog = document.getElementById('debug-log');
        
        // Add toggle button
        this.createDebugToggleButton();
        
        // Add close button functionality
        document.getElementById('close-debug-panel').addEventListener('click', () => {
            this.toggleDebugPanel();
        });
    }
    
    createDebugToggleButton() {
        // Find the navbar navigation list
        const navbarNav = document.querySelector('.navbar-nav');
        if (!navbarNav) {
            console.warn('Navbar not found, falling back to fixed positioning');
            this.createFixedDebugButton();
            return;
        }

        // Create list item for the debug button
        const listItem = document.createElement('li');
        listItem.className = 'nav-item ms-auto';
        
        // Create the debug button as a nav-link
        const toggleBtn = document.createElement('button');
        toggleBtn.id = 'debug-toggle-btn';
        toggleBtn.className = 'nav-link btn btn-link text-warning';
        toggleBtn.style.cssText = `
            border: none;
            background: none;
            font-size: 14px;
            padding: 8px 16px;
            cursor: pointer;
            text-decoration: none;
        `;
        toggleBtn.innerHTML = '<i class="fas fa-bug me-1"></i>Debug';
        toggleBtn.addEventListener('click', () => {
            this.toggleDebugPanel();
        });
        
        // Append button to list item and list item to navbar
        listItem.appendChild(toggleBtn);
        navbarNav.appendChild(listItem);
    }

    createFixedDebugButton() {
        // Fallback method for fixed positioning if navbar not found
        const toggleBtn = document.createElement('button');
        toggleBtn.id = 'debug-toggle-btn';
        toggleBtn.style.cssText = `
            position: fixed;
            top: 10px;
            right: 420px;
            background: rgba(0,123,255,0.8);
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            z-index: 9998;
        `;
        toggleBtn.innerHTML = '🔍 Debug';
        toggleBtn.addEventListener('click', () => {
            this.toggleDebugPanel();
        });
        document.body.appendChild(toggleBtn);
    }
    
    toggleDebugPanel() {
        const debugPanel = document.getElementById('debug-panel');
        const isVisible = debugPanel.style.display !== 'none';
        debugPanel.style.display = isVisible ? 'none' : 'block';
        
        const toggleBtn = document.getElementById('debug-toggle-btn');
        if (isVisible) {
            toggleBtn.innerHTML = '<i class="fas fa-bug me-1"></i>Debug';
            toggleBtn.className = 'nav-link btn btn-link text-warning';
        } else {
            toggleBtn.innerHTML = '<i class="fas fa-bug me-1"></i>Hide';
            toggleBtn.className = 'nav-link btn btn-link text-danger';
        }
    }
    
    log(message) {
        console.log(message);
        if (this.debugLog) {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.innerHTML = `<small>[${timestamp}]</small> ${message}`;
            this.debugLog.appendChild(logEntry);
            // Keep only last 10 entries
            while (this.debugLog.children.length > 10) {
                this.debugLog.removeChild(this.debugLog.firstChild);
            }
            // Scroll to bottom
            this.debugLog.scrollTop = this.debugLog.scrollHeight;
        }
    }
    
    init() {
        this.log('🚀 Initializing ModelDetailManager...');
        
        // Check if model is training or recently started
        const modelStatus = document.querySelector('[data-model-status]');
        const statusValue = modelStatus?.dataset.modelStatus;
        
        this.log(`📊 Current model status: <strong>${statusValue}</strong>`);
        this.lastKnownStatus = statusValue; // Store initial status
        
        // Start updates for training, pending, or loading states (covers recently started training)
        this.isTraining = statusValue && ['training', 'pending', 'loading'].includes(statusValue);
        
        if (this.isTraining) {
            this.log(`▶️ Starting updates for model status: <strong>${statusValue}</strong>`);
            this.startUpdates();
        } else {
            this.log(`⏸️ Model status is ${statusValue}, not starting auto-updates`);
            
            // Smart detection: Check if this might be a recently created model that will start training soon
            const modelElement = document.querySelector('[data-model-id]');
            const modelCreatedAt = modelElement?.dataset.modelCreatedAt;
            
            if (modelCreatedAt) {
                const createdTime = new Date(modelCreatedAt);
                const now = new Date();
                const minutesSinceCreation = (now - createdTime) / (1000 * 60);
                
                this.log(`⏰ Model created <strong>${minutesSinceCreation.toFixed(1)}</strong> minutes ago`);
                
                // If model was created within the last 10 minutes, poll for status changes
                if (minutesSinceCreation < 10) {
                    this.log(`👀 Model is recent, watching for training start`);
                    this.startTrainingWatch();
                }
            }
        }
        
        this.setupEventListeners();
        this.requestNotificationPermission();
        this.log('✅ Initialization complete');
    }
    
    requestNotificationPermission() {
        // Request browser notification permission for training completion alerts
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission().then(permission => {
                if (permission === 'granted') {
                    this.log(`🔔 Browser notifications enabled`);
                } else {
                    this.log(`🔕 Browser notifications denied`);
                }
            });
        }
    }
    
    startTrainingWatch() {
        // Watch for training to start for up to 3 minutes with enhanced status detection
        let watchAttempts = 0;
        const maxAttempts = 36; // 36 * 5 seconds = 3 minutes
        
        this.log(`👀 Starting enhanced training watch (max ${maxAttempts * 5}s)`);
        
        const watchInterval = setInterval(() => {
            watchAttempts++;
            
            fetch(`/ml/model/${this.modelId}/progress/`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success' && data.model_status) {
                        const currentStatus = data.model_status;
                        
                        // Enhanced status transition detection
                        if (data.status_changed && data.status_transition) {
                            this.log(`🚀 Status transition detected: <strong>${data.status_transition}</strong>`);
                            this.statusTransitionDetected = true;
                        }
                        
                        // Check if training has started
                        if (['training', 'loading'].includes(currentStatus)) {
                            this.log(`🚀 Training started! Status: <strong>${currentStatus}</strong>`);
                            clearInterval(watchInterval);
                            this.isTraining = true;
                            this.lastKnownStatus = currentStatus;
                            this.startUpdates();
                        } else if (currentStatus !== this.lastKnownStatus) {
                            // Log any status changes even if not training yet
                            this.log(`📊 Status change: <strong>${this.lastKnownStatus}</strong> → <strong>${currentStatus}</strong>`);
                            this.lastKnownStatus = currentStatus;
                            
                            // If status changed to completed or failed, stop watching
                            if (['completed', 'failed', 'stopped'].includes(currentStatus)) {
                                this.log(`🛑 Training ended with status: <strong>${currentStatus}</strong>`);
                                clearInterval(watchInterval);
                                // Reload page to show final state
                                setTimeout(() => location.reload(), 1500);
                            }
                        } else if (watchAttempts >= maxAttempts) {
                            this.log('⏰ Training watch timeout, stopping');
                            clearInterval(watchInterval);
                        }
                    }
                })
                .catch(error => {
                    console.error('Training watch error:', error);
                    if (watchAttempts >= maxAttempts) {
                        clearInterval(watchInterval);
                    }
                });
        }, 5000); // Check every 5 seconds
    }
    
    setupEventListeners() {
        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to stop training? This cannot be undone.')) {
                    this.stopTraining();
                }
            });
        }
        
        if (this.elements.manualRefreshBtn) {
            this.elements.manualRefreshBtn.addEventListener('click', () => {
                this.manualRefresh();
            });
        }
        
        window.addEventListener('beforeunload', () => {
            this.stopUpdates();
        });
    }
    
    startUpdates() {
        if (this.updateInterval) return;
        
        // Use faster interval for pending→training transitions
        const intervalMs = this.lastKnownStatus === 'pending' ? 1000 : this.updateIntervalMs;
        
        this.log(`🔄 Starting updates with interval: <strong>${intervalMs}ms</strong>`);
        this.updateProgress();
        this.updateInterval = setInterval(() => {
            this.updateProgress();
        }, intervalMs);
        
        this.showLiveIndicator();
        
        // If we're in pending state, check more frequently initially
        if (this.lastKnownStatus === 'pending') {
            this.log(`⚡ Using fast polling for pending status`);
            setTimeout(() => {
                if (this.updateInterval && this.lastKnownStatus !== 'pending') {
                    // Switch to normal interval once we're out of pending
                    this.log(`🔄 Switching to normal polling interval`);
                    clearInterval(this.updateInterval);
                    this.updateInterval = setInterval(() => {
                        this.updateProgress();
                    }, this.updateIntervalMs);
                }
            }, 30000); // Check after 30 seconds
        }
    }
    
    stopUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            console.log('Stopped model detail updates');
        }
        this.hideLiveIndicator();
    }
    
    showLiveIndicator() {
        if (this.elements.updateStatus) {
            this.elements.updateStatus.innerHTML = '<i class="fas fa-circle text-success" style="font-size: 0.5rem;"></i> Live';
            this.elements.updateStatus.style.display = 'inline-block';
        }
    }
    
    hideLiveIndicator() {
        if (this.elements.updateStatus) {
            this.elements.updateStatus.style.display = 'none';
        }
    }
    
    setLoadingIndicator() {
        if (this.elements.updateStatus) {
            this.elements.updateStatus.innerHTML = '<i class="fas fa-circle text-warning" style="font-size: 0.5rem; animation: pulse 1s infinite;"></i> Updating...';
        }
    }
    
    setErrorIndicator() {
        if (this.elements.updateStatus) {
            this.elements.updateStatus.innerHTML = '<i class="fas fa-circle text-danger" style="font-size: 0.5rem;"></i> Error';
        }
    }
    
    updateProgress() {
        this.setLoadingIndicator();
        this.log(`📡 Fetching progress for model <strong>${this.modelId}</strong>`);
        
        fetch(`/ml/model/${this.modelId}/progress/`)
            .then(response => {
                this.log(`📊 Progress API response status: <strong>${response.status}</strong>`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                const statusInfo = data.model_status ? `Status = <strong>${data.model_status}</strong>` : 'No status';
                const transitionInfo = data.status_transition ? `, Transition: <strong>${data.status_transition}</strong>` : '';
                this.log(`📦 Progress data received: ${statusInfo}${transitionInfo}`);
                
                if (data.status === 'success') {
                    this.handleSuccessfulUpdate(data);
                } else {
                    this.log(`❌ API returned error status: ${JSON.stringify(data)}`);
                    this.setErrorIndicator();
                }
            })
            .catch(error => {
                this.log(`🚨 Progress update error: <strong>${error.message}</strong>`);
                this.setErrorIndicator();
                
                // If error persists, reduce update frequency to avoid spam
                if (error.message.includes('404') || error.message.includes('500')) {
                    this.log(`⚠️ Server error detected, reducing update frequency`);
                    if (this.updateInterval) {
                        clearInterval(this.updateInterval);
                        this.updateInterval = setInterval(() => {
                            this.updateProgress();
                        }, this.updateIntervalMs * 3); // 3x slower
                    }
                }
            });
    }
    
    handleSuccessfulUpdate(data) {
        // Enhanced status transition detection
        const currentModelStatus = data.model_status;
        const previousStatus = this.lastKnownStatus;
        
        // Detect status changes with improved logic
        if (currentModelStatus !== previousStatus) {
            this.log(`📊 Status transition: <strong>${previousStatus}</strong> → <strong>${currentModelStatus}</strong>`);
            this.lastKnownStatus = currentModelStatus;
            this.statusTransitionDetected = true;
            
            // Update page title to reflect status
            document.title = `${currentModelStatus.toUpperCase()} - Model ${this.modelId}`;
        }
        
        // Log API status change detection
        if (data.status_changed && data.status_transition) {
            this.log(`🔄 API detected transition: <strong>${data.status_transition}</strong>`);
        }
        
        // Check for training completion or termination
        if (currentModelStatus && !['training', 'pending', 'loading'].includes(currentModelStatus)) {
            this.log(`🎉 Training completed with status: <strong>${currentModelStatus}</strong>, stopping updates`);
            this.stopUpdates();
            
            // Show completion notification
            this.showCompletionNotification(currentModelStatus);
            
            // Reload page after a short delay to show final state
            this.log(`🔄 Reloading page in 3 seconds to show final state...`);
            setTimeout(() => {
                location.reload();
            }, 3000);
            return;
        }
        
        // Continue with normal updates
        this.updateProgressBars(data.progress);
        this.updateMetrics(data.metrics, data.progress);
        this.updateLastUpdateTime();
        this.showLiveIndicator();
        
        // Refresh training images on epoch changes
        if (data.progress && data.progress.current_epoch) {
            this.refreshTrainingImages();
        }
        
        // Legacy status_changed check (for backward compatibility)
        if (data.status_changed && !this.statusTransitionDetected) {
            setTimeout(() => {
                location.reload();
            }, 2000);
        }
    }
    
    showCompletionNotification(status) {
        const statusText = status === 'completed' ? 'Successfully Completed' : 
                          status === 'failed' ? 'Failed' : 
                          status === 'stopped' ? 'Stopped' : 'Finished';
        
        const statusColor = status === 'completed' ? 'success' : 
                           status === 'failed' ? 'danger' : 'warning';
        
        this.log(`🎯 Training ${statusText}!`);
        
        // Show browser notification if supported
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(`Training ${statusText}`, {
                body: `Model ${this.modelId} training has ${statusText.toLowerCase()}`,
                icon: '/static/img/favicon.ico'
            });
        }
        
        // Update any status badges on the page
        const statusBadges = document.querySelectorAll('.badge');
        statusBadges.forEach(badge => {
            if (badge.textContent.toLowerCase().includes('training') || 
                badge.textContent.toLowerCase().includes('pending')) {
                badge.textContent = statusText;
                badge.className = `badge bg-${statusColor}`;
            }
        });
    }
    
    updateProgressBars(progress) {
        if (!progress) return;
        
        if (this.elements.progressBar) {
            this.elements.progressBar.style.width = progress.percentage + '%';
            this.elements.progressBar.textContent = `Epoch ${progress.current_epoch}/${progress.total_epochs} (${progress.percentage.toFixed(1)}%)`;
            this.elements.progressBar.setAttribute('aria-valuenow', progress.current_epoch);
        }
        
        if (this.elements.batchProgressBar && progress.batch_progress_percentage !== undefined) {
            this.elements.batchProgressBar.style.width = progress.batch_progress_percentage + '%';
            this.elements.batchProgressBar.setAttribute('aria-valuenow', progress.current_batch);
        }
    }
    
    updateMetrics(metrics, progress) {
        if (!metrics) return;
        
        const updateMetric = (id, value, isInteger = false) => {
            const element = document.getElementById(id);
            if (element && value !== null && value !== undefined) {
                const formattedValue = isInteger ? value.toString() : 
                                     (typeof value === 'number' ? value.toFixed(4) : value);
                
                if (element.textContent !== formattedValue) {
                    element.textContent = formattedValue;
                    element.style.backgroundColor = '#e7f3ff';
                    setTimeout(() => {
                        element.style.backgroundColor = '';
                    }, 500);
                }
            }
        };
        
        if (progress) {
            updateMetric('current-epoch', progress.current_epoch, true); // Integer formatting
        }
        updateMetric('train-loss', metrics.train_loss);
        updateMetric('train-dice', metrics.train_dice);
        updateMetric('val-loss', metrics.val_loss);
        updateMetric('val-dice', metrics.val_dice);
        updateMetric('best-val-dice', metrics.best_val_dice);
    }
    
    updateLastUpdateTime() {
        if (this.elements.lastUpdateTime) {
            const now = new Date();
            this.elements.lastUpdateTime.textContent = `Last updated: ${now.toLocaleTimeString()}`;
        }
    }
    
    manualRefresh() {
        if (this.elements.manualRefreshBtn) {
            this.elements.manualRefreshBtn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Refreshing...';
        }
        
        this.updateProgress();
        
        setTimeout(() => {
            if (this.elements.manualRefreshBtn) {
                this.elements.manualRefreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Now';
            }
        }, 1000);
    }
    
    stopTraining() {
        const formData = new FormData();
        formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]').value);
        
        fetch(`/ml/model/${this.modelId}/stop/`, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                if (this.elements.stopBtn) {
                    this.elements.stopBtn.disabled = true;
                    this.elements.stopBtn.textContent = 'Stop Requested';
                }
                this.showAlert('info', 'Stop request sent. Training will stop after current epoch.');
            } else {
                this.showAlert('danger', data.message || 'Failed to stop training');
            }
        })
        .catch(error => {
            console.error('Error stopping training:', error);
            this.showAlert('danger', 'Error stopping training');
        });
    }
    
    refreshTrainingImages() {
        const trainingSamplesCard = document.querySelector('.training-samples-container');
        if (!trainingSamplesCard) return;
        
        fetch(window.location.href, {
            headers: { 'X-Requested-With': 'XMLHttpRequest' }
        })
        .then(response => response.text())
        .then(html => {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const newContainer = doc.querySelector('.training-samples-container');
            
            if (newContainer) {
                trainingSamplesCard.innerHTML = newContainer.innerHTML;
                console.log('Training images refreshed');
            }
        })
        .catch(error => {
            console.error('Error refreshing training images:', error);
        });
    }
    
    // MLflow Registry Management Methods
    async registerModel(modelId) {
        if (!confirm('Register this model in MLflow Model Registry?')) {
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]').value);
            
            const response = await fetch(`/ml/model/${modelId}/registry/register/`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showAlert('success', data.message);
                location.reload();
            } else {
                this.showAlert('danger', data.message);
            }
        } catch (error) {
            console.error('Registry error:', error);
            this.showAlert('danger', 'Failed to register model');
        }
    }
    
    async syncRegistryInfo(modelId) {
        try {
            const response = await fetch(`/ml/model/${modelId}/registry/sync/`, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
                }
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showAlert('success', data.message);
                if (data.changes && data.changes.stage_changed) {
                    const stageBadge = document.getElementById('registry-stage');
                    if (stageBadge) {
                        stageBadge.textContent = data.changes.new_stage || 'None';
                    }
                }
            } else {
                this.showAlert('danger', data.message);
            }
        } catch (error) {
            console.error('Sync error:', error);
            this.showAlert('danger', 'Failed to sync registry info');
        }
    }
    
    // Logs Modal Methods
    openLogsModal() {
        console.log('Opening logs modal...');
        try {
            const modalElement = document.getElementById('logsModal');
            if (!modalElement) {
                console.error('Logs modal element not found');
                return;
            }
            
            if (typeof bootstrap === 'undefined') {
                console.error('Bootstrap is not loaded');
                alert('Unable to open logs modal: Bootstrap library not loaded');
                return;
            }
            
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
            
            // Load logs when modal opens
            this.loadModalLogs();
            console.log('Logs modal opened successfully');
        } catch (error) {
            console.error('Error opening logs modal:', error);
            alert('Error opening logs modal: ' + error.message);
        }
    }
    
    async loadModalLogs() {
        const logsContent = document.getElementById('modal-logs-content');
        const totalLinesSpan = document.getElementById('modal-total-lines');
        
        if (!logsContent) {
            console.error('Modal logs content element not found');
            return;
        }
        
        logsContent.innerHTML = '<div class="text-center py-5"><i class="fas fa-spinner fa-spin fa-2x text-primary"></i><p class="mt-2">Loading logs...</p></div>';
        
        try {
            const response = await fetch(`/ml/model/${this.modelId}/logs/`, {
                method: 'GET',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/json',
                },
                credentials: 'same-origin'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            let logsToDisplay = [];
            
            if (data.logs && Array.isArray(data.logs)) {
                logsToDisplay = data.logs.filter(line => line && line.trim() !== '');
            }
            
            // Filter out DEBUG log lines
            logsToDisplay = logsToDisplay.filter(line => {
                const text = typeof line === 'object' && line.content ? line.content : String(line);
                return !/^\d{4}-\d{2}-\d{2}.*\bDEBUG\b/.test(text);
            });
            
            if (logsToDisplay.length > 0) {
                const logHTML = logsToDisplay.map(line => {
                    const text = typeof line === 'object' && line.content ? line.content : String(line);
                    const escapedLine = text.replace(/&/g, '&amp;')
                                           .replace(/</g, '&lt;')
                                           .replace(/>/g, '&gt;');
                    return `<div class="log-entry">${escapedLine}</div>`;
                }).join('');
                logsContent.innerHTML = logHTML;
                if (totalLinesSpan) totalLinesSpan.textContent = logsToDisplay.length;
            } else {
                logsContent.innerHTML = '<div class="text-center py-5"><i class="fas fa-info-circle fa-2x text-warning"></i><p class="mt-2">No logs available yet.</p></div>';
                if (totalLinesSpan) totalLinesSpan.textContent = '0';
            }
            logsContent.scrollTop = logsContent.scrollHeight;
        } catch (error) {
            console.error('Error loading logs:', error);
            logsContent.innerHTML = `<div class="text-center py-5"><i class="fas fa-exclamation-triangle fa-2x text-danger"></i><p class="mt-2">Failed to load logs: ${error.message}</p><p class="text-muted">Please check browser console for details.</p></div>`;
        }
    }
    
    // Alert system
    showAlert(type, message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show mt-3`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert after the header
        const header = document.querySelector('h1').parentElement;
        header.insertAdjacentElement('afterend', alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.remove();
            }
        }, 5000);
    }
}
