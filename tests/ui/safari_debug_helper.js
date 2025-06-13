// Safari Debug Helper - wklej ten kod w Safari Web Inspector Console
// Pomoże zdiagnozować problemy z dropdown

console.log('🔍 Safari Dropdown Debug Helper');
console.log('================================');

// 1. Sprawdź czy Safari fixes są załadowane
console.log('1. Sprawdzanie czy Safari fixes są aktywne...');
const safariStyle = document.querySelector('style[data-safari-fixes]') || 
                   Array.from(document.querySelectorAll('style')).find(s => 
                       s.textContent.includes('Safari-specific dropdown fixes')
                   );
console.log('   Safari CSS fixes:', safariStyle ? '✅ Załadowane' : '❌ Nie znalezione');

// 2. Sprawdź dropdowns
console.log('2. Sprawdzanie dropdown elementów...');
const dropdowns = document.querySelectorAll('.custom-dropdown');
console.log(`   Znalezione dropdowns: ${dropdowns.length}`);

dropdowns.forEach((dropdown, index) => {
    console.log(`   Dropdown ${index + 1}:`);
    const toggle = dropdown.querySelector('.custom-dropdown-toggle');
    const menu = dropdown.querySelector('.custom-dropdown-menu');
    console.log(`     - Toggle: ${toggle ? '✅' : '❌'}`);
    console.log(`     - Menu: ${menu ? '✅' : '❌'}`);
    console.log(`     - Classes: ${dropdown.className}`);
});

// 3. Test funkcji kliknięcia
console.log('3. Testowanie pierwszego dropdown...');
if (dropdowns.length > 0) {
    const firstDropdown = dropdowns[0];
    const toggle = firstDropdown.querySelector('.custom-dropdown-toggle');
    const menu = firstDropdown.querySelector('.custom-dropdown-menu');
    
    if (toggle && menu) {
        console.log('   Symulowanie kliknięcia...');
        
        // Sprawdź stan przed
        const beforeState = {
            dropdownHasShow: firstDropdown.classList.contains('show'),
            menuHasShow: menu.classList.contains('show'),
            menuDisplay: window.getComputedStyle(menu).display,
            menuVisibility: window.getComputedStyle(menu).visibility,
            menuOpacity: window.getComputedStyle(menu).opacity
        };
        console.log('   Stan przed kliknięciem:', beforeState);
        
        // Symuluj kliknięcie
        const clickEvent = new MouseEvent('click', {
            bubbles: true,
            cancelable: true,
            view: window
        });
        toggle.dispatchEvent(clickEvent);
        
        // Sprawdź stan po krótkim opóźnieniu
        setTimeout(() => {
            const afterState = {
                dropdownHasShow: firstDropdown.classList.contains('show'),
                menuHasShow: menu.classList.contains('show'),
                menuDisplay: window.getComputedStyle(menu).display,
                menuVisibility: window.getComputedStyle(menu).visibility,
                menuOpacity: window.getComputedStyle(menu).opacity
            };
            console.log('   Stan po kliknięciu:', afterState);
            
            // Sprawdź czy się zmieniło
            const changed = JSON.stringify(beforeState) !== JSON.stringify(afterState);
            console.log(`   Czy dropdown zareagował: ${changed ? '✅' : '❌'}`);
            
            if (changed && afterState.dropdownHasShow) {
                console.log('   ✅ Dropdown się otworzył!');
                
                // Test menu items
                const menuItems = menu.querySelectorAll('.custom-dropdown-item');
                console.log(`   Elementy menu: ${menuItems.length}`);
                
                menuItems.forEach((item, i) => {
                    console.log(`     Item ${i + 1}: "${item.textContent.trim()}"`);
                });
            } else {
                console.log('   ❌ Dropdown nie zareagował');
            }
        }, 100);
    }
}

// Helper function do ręcznego testowania
window.testSafariDropdown = function() {
    console.log('🧪 Ręczny test dropdown...');
    const dropdown = document.querySelector('.custom-dropdown');
    if (dropdown) {
        const toggle = dropdown.querySelector('.custom-dropdown-toggle');
        const menu = dropdown.querySelector('.custom-dropdown-menu');
        
        // Force open
        dropdown.classList.add('show');
        menu.classList.add('show');
        menu.style.display = 'block';
        menu.style.visibility = 'visible';
        menu.style.opacity = '1';
        
        console.log('Dropdown force opened - sprawdź czy jest widoczny');
        
        setTimeout(() => {
            dropdown.classList.remove('show');
            menu.classList.remove('show');
            menu.style.display = '';
            menu.style.visibility = '';
            menu.style.opacity = '';
            console.log('Dropdown force closed');
        }, 3000);
    }
};

console.log('================================');
console.log('💡 Dostępne komendy:');
console.log('   testSafariDropdown() - test ręczny');
console.log('================================');