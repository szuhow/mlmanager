#!/bin/bash
# filepath: /Users/rafalszulinski/Desktop/developing/IVES/coronary/git/mlmanager/scripts/test-input.sh

echo "🧪 Testing input functionality..."
echo ""

echo "Test 1: Simple read"
echo -n "Type 'y' and press Enter: "
read -r answer1
echo "You typed: '$answer1'"
echo ""

echo "Test 2: Read with prompt"
read -p "Type 'yes' and press Enter: " answer2
echo "You typed: '$answer2'"
echo ""

echo "Test 3: Single character read"
echo -n "Type any key (no Enter needed): "
read -n 1 answer3
echo ""
echo "You typed: '$answer3'"
echo ""

if [[ $answer1 == "y" ]] && [[ $answer2 == "yes" ]]; then
    echo "✅ Input functionality works correctly!"
else
    echo "❌ Input functionality has issues"
    echo "  answer1: '$answer1' (expected: 'y')"
    echo "  answer2: '$answer2' (expected: 'yes')"
fi
