// Safari-specific dropdown fixes
// This file contains fixes for dropdown compatibility issues in Safari

(function() {
    'use strict';
    
    // Detect Safari (including mobile Safari)
    const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent) || 
                     /iPad|iPhone|iPod/.test(navigator.userAgent) ||
                     (navigator.vendor && navigator.vendor.indexOf('Apple') > -1);
    
    if (!isSafari) {
        console.log('Not Safari - skipping Safari-specific fixes');
        return; // Only apply fixes in Safari
    }
    
    console.log('Applying Safari-specific dropdown fixes for:', navigator.userAgent);
    
    // Add Safari-specific CSS fixes
    const safariCSS = `
        /* Safari-specific dropdown fixes */
        .custom-dropdown {
            /* Force hardware acceleration */
            -webkit-transform: translateZ(0);
            transform: translateZ(0);
            -webkit-backface-visibility: hidden;
            backface-visibility: hidden;
            /* Improve pointer events */
            pointer-events: auto;
        }
        
        .custom-dropdown-menu {
            /* Disable iOS momentum scrolling issues */
            -webkit-overflow-scrolling: touch;
            /* Force repainting */
            -webkit-transform: translate3d(0,0,0);
            transform: translate3d(0,0,0);
            /* Improve rendering */
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            /* CRITICAL: Use fixed positioning to escape table stacking context */
            position: fixed !important;
            z-index: 999999 !important;
            /* Fix Safari rendering */
            will-change: transform, opacity;
            /* Ensure proper pointer events */
            pointer-events: auto;
            /* Force Safari to respect z-index */
            -webkit-transform-style: preserve-3d;
            transform-style: preserve-3d;
        }
        
        .custom-dropdown-menu.show {
            /* Force visibility with maximum z-index */
            display: block !important;
            opacity: 1 !important;
            visibility: visible !important;
            /* Prevent Safari from hiding the menu */
            -webkit-transform: translate3d(0,0,0) !important;
            transform: translate3d(0,0,0) !important;
            /* CRITICAL: Maximum z-index for Safari */
            z-index: 99999 !important;
            /* Force new stacking context */
            position: absolute !important;
            /* Ensure it's above everything */
            -webkit-transform-style: preserve-3d !important;
            transform-style: preserve-3d !important;
        }
        
        .custom-dropdown-toggle {
            /* Improve button rendering */
            -webkit-appearance: none;
            appearance: none;
            /* Force repainting */
            -webkit-transform: translateZ(0);
            transform: translateZ(0);
            /* Fix Safari button issues */
            cursor: pointer;
            outline: none;
            /* Disable Safari tap highlighting */
            -webkit-tap-highlight-color: transparent;
            /* Ensure touch events work */
            touch-action: manipulation;
        }
        
        .custom-dropdown-item {
            /* Fix hover states in Safari */
            -webkit-tap-highlight-color: transparent;
            /* Improve text rendering */
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            /* Ensure proper cursor */
            cursor: pointer;
            /* Improve touch response */
            touch-action: manipulation;
        }
        
        /* Fix table positioning issues in Safari */
        .table tbody tr {
            position: relative !important;
            /* Force new stacking context */
            -webkit-transform: translateZ(0);
            transform: translateZ(0);
            /* Default z-index for table rows */
            z-index: 1;
        }
        
        .table tbody tr.dropdown-open {
            /* VERY HIGH z-index when dropdown is open */
            z-index: 100000 !important;
            /* Force higher stacking */
            position: relative !important;
            /* Create isolated stacking context */
            -webkit-transform-style: preserve-3d !important;
            transform-style: preserve-3d !important;
        }
        
        /* Ensure table cells don't interfere */
        .table tbody tr td {
            position: relative;
            z-index: auto;
        }
        
        /* Special handling for actions cell */
        .table tbody tr.dropdown-open td.actions-cell {
            z-index: 100001 !important;
            position: relative !important;
        }
        
        /* Safari-specific animation fixes */
        .custom-dropdown-toggle::after {
            /* Disable problematic transitions in Safari */
            transition: none !important;
            -webkit-transition: none !important;
        }
        
        /* Fix Safari rendering glitches */
        .actions-cell {
            /* Force new stacking context */
            position: relative;
            z-index: 1;
            /* Improve rendering */
            -webkit-transform: translateZ(0);
            transform: translateZ(0);
        }
        
        /* Fix Safari table overflow issues */
        .table-responsive {
            overflow: visible !important;
            -webkit-overflow-scrolling: touch;
        }
        
        /* Force Safari to create proper stacking contexts */
        .table {
            /* Prevent table from creating its own stacking context */
            position: static !important;
        }
        
        /* Ensure Safari respects z-index on table elements */
        .table tbody {
            position: relative;
            z-index: 1;
        }
        
        /* Force dropdown to appear above everything in Safari */
        .custom-dropdown-menu.show {
            /* Use fixed positioning if absolute fails in Safari table */
            position: fixed !important;
            /* Calculate position relative to viewport */
        }
        
        /* Additional Safari-specific fixes */
        @supports (-webkit-touch-callout: none) {
            .custom-dropdown-menu {
                /* iOS Safari specific fixes */
                -webkit-user-select: none;
                user-select: none;
                /* Force iOS to respect z-index */
                -webkit-transform: translate3d(0,0,1px) !important;
                transform: translate3d(0,0,1px) !important;
            }
            
            .custom-dropdown-toggle {
                /* Prevent zoom on double tap */
                touch-action: manipulation;
                -webkit-user-select: none;
                user-select: none;
            }
            
            /* Force high z-index on iOS */
            .table tbody tr.dropdown-open {
                z-index: 999999 !important;
                -webkit-transform: translate3d(0,0,1px) !important;
                transform: translate3d(0,0,1px) !important;
            }
        }
    `;
    
    // Inject Safari-specific CSS
    const style = document.createElement('style');
    style.setAttribute('data-safari-fixes', 'true');
    style.textContent = safariCSS;
    document.head.appendChild(style);
    
    // Wait for the main dropdown initialization to complete
    let initAttempts = 0;
    const maxAttempts = 50; // 5 seconds max wait
    
    function applySafariFixes() {
        // Check if main dropdown system is initialized
        const dropdowns = document.querySelectorAll('.custom-dropdown');
        if (dropdowns.length === 0 && initAttempts < maxAttempts) {
            initAttempts++;
            setTimeout(applySafariFixes, 100);
            return;
        }
        
        console.log('Applying Safari dropdown fixes to', dropdowns.length, 'dropdowns');
        
        // Use MutationObserver to watch for dropdown state changes
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                    const target = mutation.target;
                    if (target.classList.contains('custom-dropdown')) {
                        const menu = target.querySelector('.custom-dropdown-menu');
                        const parentRow = target.closest('tr');
                        
                        if (menu && target.classList.contains('show')) {
                            // Safari-specific fixes when dropdown opens
                            console.log('🔧 Safari: Opening dropdown with fixed positioning');
                            setTimeout(() => {
                                // Force display first
                                menu.style.display = 'block';
                                menu.style.visibility = 'visible';
                                menu.style.opacity = '1';
                                
                                // Get toggle position relative to viewport
                                const toggle = target.querySelector('.custom-dropdown-toggle');
                                if (toggle) {
                                    const toggleRect = toggle.getBoundingClientRect();
                                    console.log('🔧 Safari: Toggle position:', toggleRect);
                                    
                                    // Use fixed positioning for Safari to escape table stacking context
                                    menu.style.position = 'fixed';
                                    menu.style.top = (toggleRect.bottom) + 'px';
                                    menu.style.left = (toggleRect.left) + 'px';
                                    menu.style.zIndex = '999999';
                                    
                                    console.log('🔧 Safari: Menu positioned at:', {
                                        top: menu.style.top,
                                        left: menu.style.left,
                                        position: menu.style.position,
                                        zIndex: menu.style.zIndex
                                    });
                                    
                                    // Check if menu would go off screen
                                    setTimeout(() => {
                                        const menuRect = menu.getBoundingClientRect();
                                        console.log('🔧 Safari: Menu dimensions:', menuRect);
                                        
                                        // Adjust horizontal position if off-screen
                                        if (toggleRect.left + menuRect.width > window.innerWidth) {
                                            menu.style.left = (toggleRect.right - menuRect.width) + 'px';
                                            console.log('🔧 Safari: Adjusted left position to:', menu.style.left);
                                        }
                                        
                                        // Adjust vertical position if off-screen
                                        if (toggleRect.bottom + menuRect.height > window.innerHeight) {
                                            menu.style.top = (toggleRect.top - menuRect.height) + 'px';
                                            console.log('🔧 Safari: Adjusted top position to:', menu.style.top);
                                        }
                                    }, 1);
                                }
                                
                                // Force Safari repaint
                                menu.style.webkitTransform = 'translate3d(0,0,0)';
                                menu.style.transform = 'translate3d(0,0,0)';
                            }, 1);
                        } else if (menu && !target.classList.contains('show')) {
                            console.log('🔧 Safari: Closing dropdown and resetting position');
                            // Reset positioning when dropdown closes
                            menu.style.position = '';
                            menu.style.top = '';
                            menu.style.left = '';
                            menu.style.zIndex = '';
                        }
                    }
                }
            });
        });
        
        // Observe each dropdown for class changes
        dropdowns.forEach(dropdown => {
            observer.observe(dropdown, { 
                attributes: true, 
                attributeFilter: ['class']
            });
        });
        
        // Add touch feedback for Safari
        document.querySelectorAll('.custom-dropdown-toggle').forEach(toggle => {
            if (toggle.hasAttribute('data-safari-enhanced')) return;
            toggle.setAttribute('data-safari-enhanced', 'true');
            
            toggle.addEventListener('touchstart', function(e) {
                this.style.backgroundColor = '#5a6268';
                this.style.transform = 'scale(0.98)';
            }, { passive: true });
            
            toggle.addEventListener('touchend', function(e) {
                setTimeout(() => {
                    this.style.backgroundColor = '';
                    this.style.transform = '';
                }, 150);
            }, { passive: true });
            
            toggle.addEventListener('touchcancel', function(e) {
                this.style.backgroundColor = '';
                this.style.transform = '';
            }, { passive: true });
        });
        
        // Enhanced outside click handling for Safari
        let outsideClickHandler = function(e) {
            if (!e.target.closest('.custom-dropdown')) {
                document.querySelectorAll('.custom-dropdown.show').forEach(dropdown => {
                    dropdown.classList.remove('show');
                    const menu = dropdown.querySelector('.custom-dropdown-menu');
                    if (menu) {
                        menu.classList.remove('show');
                    }
                });
            }
        };
        
        // Remove existing outside click handlers to avoid duplicates
        document.removeEventListener('click', outsideClickHandler, true);
        document.addEventListener('click', outsideClickHandler, true); // Use capture phase
        
        // Prevent menu from closing when clicking inside
        document.querySelectorAll('.custom-dropdown-menu').forEach(menu => {
            menu.addEventListener('click', function(e) {
                e.stopPropagation();
            });
            
            // Fix Safari scroll issues
            menu.addEventListener('touchmove', function(e) {
                e.stopPropagation();
            }, { passive: true });
        });
    }
    
    // Start applying fixes when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applySafariFixes);
    } else {
        applySafariFixes();
    }
    
    // Additional Safari-specific fixes
    window.addEventListener('resize', function() {
        // Close all dropdowns on resize to prevent positioning issues
        document.querySelectorAll('.custom-dropdown.show').forEach(dropdown => {
            dropdown.classList.remove('show');
            const menu = dropdown.querySelector('.custom-dropdown-menu');
            if (menu) {
                menu.classList.remove('show');
            }
        });
    }, { passive: true });
    
    // Fix Safari scrolling issues
    window.addEventListener('scroll', function() {
        // Close dropdowns on scroll for better UX in Safari
        document.querySelectorAll('.custom-dropdown.show').forEach(dropdown => {
            dropdown.classList.remove('show');
            const menu = dropdown.querySelector('.custom-dropdown-menu');
            if (menu) {
                menu.classList.remove('show');
            }
        });
    }, { passive: true });
    
})();
