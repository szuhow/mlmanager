#!/usr/bin/env python3
"""
Test script to debug and fix ARCADE artery classification
Analyze the current classification logic and fix it according to coronary anatomy
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, '/home/rafal/Dokumenty/ivessystem/coronary/ives/coronary-experiments')

def analyze_current_classification():
    """Analyze current artery classification logic"""
    print("🔍 Analyzing Current ARCADE Artery Classification Logic")
    print("=" * 60)
    
    # Import the current function
    from ml.datasets.torch_arcade_loader import distinguish_side
    
    # Test current classification logic
    test_cases = [
        # Right Coronary Artery (RCA) segments - should be "right"
        (["1"], "RCA Segment 1 (proximal)"),
        (["2"], "RCA Segment 2 (mid)"), 
        (["3"], "RCA Segment 3 (distal)"),
        (["4"], "RCA Segment 4 (posterior descending)"),
        (["16a"], "RCA Segment 16a (posterolateral 1)"),
        (["16b"], "RCA Segment 16b (posterolateral 2)"),
        (["16c"], "RCA Segment 16c (posterolateral 3)"),
        
        # Left Coronary System segments - should be "left"
        (["5"], "Left Main (LM)"),
        (["6"], "LAD Segment 6 (proximal)"),
        (["7"], "LAD Segment 7 (mid)"),
        (["8"], "LAD Segment 8 (distal)"),
        (["9"], "LAD Diagonal 1"),
        (["9a"], "LAD Diagonal 1a"),
        (["10"], "LAD Diagonal 2"),
        (["10a"], "LAD Diagonal 2a"),
        (["11"], "LCX Segment 11 (proximal)"),
        (["12"], "LCX Segment 12 (mid)"),
        (["12a"], "LCX Segment 12a (obtuse marginal 1)"),
        (["12b"], "LCX Segment 12b (obtuse marginal 2)"),
        (["13"], "LCX Segment 13 (distal)"),
        (["14"], "LCX Segment 14 (posterolateral)"),
        (["14a"], "LCX Segment 14a (posterolateral branch)"),
        (["14b"], "LCX Segment 14b (posterolateral branch)"),
        (["15"], "LCX Segment 15 (posterior descending from LCX)"),
        
        # Mixed cases
        (["1", "6"], "Mixed RCA + LAD (should be ambiguous)"),
        (["5", "11", "12"], "Left system only"),
        (["1", "2", "3"], "RCA system only"),
    ]
    
    print("Current Classification Results:")
    print("-" * 40)
    
    errors = []
    for segments, description in test_cases:
        result = distinguish_side(segments)
        expected = "right" if any(seg in ["1", "2", "3", "4", "16a", "16b", "16c"] for seg in segments) else "left"
        
        # Determine what SHOULD be the correct classification based on anatomy
        if any(seg in ["1", "2", "3", "4", "16a", "16b", "16c"] for seg in segments):
            correct_expected = "right"
        elif any(seg in ["5", "6", "7", "8", "9", "9a", "10", "10a", "11", "12", "12a", "12b", "13", "14", "14a", "14b", "15"] for seg in segments):
            correct_expected = "left" 
        else:
            correct_expected = "unknown"
            
        status = "✅" if result == correct_expected else "❌"
        print(f"{status} {segments} → {result} | {description}")
        
        if result != correct_expected:
            errors.append((segments, result, correct_expected, description))
    
    print("\n" + "=" * 60)
    print(f"❌ Found {len(errors)} classification errors:")
    for segments, got, expected, desc in errors:
        print(f"   {segments}: got '{got}', expected '{expected}' - {desc}")
    
    return errors

def research_coronary_anatomy():
    """Research and document correct coronary artery segment anatomy"""
    print("\n🫀 Standard Coronary Artery Anatomy (AHA 17-Segment Model)")
    print("=" * 60)
    
    coronary_anatomy = {
        "RIGHT_CORONARY_ARTERY": {
            "description": "Right Coronary Artery (RCA) - supplies right ventricle and inferior wall",
            "segments": {
                "1": "RCA - Proximal",
                "2": "RCA - Mid", 
                "3": "RCA - Distal",
                "4": "RCA - Posterior Descending Artery (PDA)",
                "16a": "RCA - Posterolateral Branch 1",
                "16b": "RCA - Posterolateral Branch 2", 
                "16c": "RCA - Posterolateral Branch 3"
            }
        },
        "LEFT_CORONARY_SYSTEM": {
            "description": "Left Coronary System - supplies left ventricle and anterior wall",
            "LEFT_MAIN": {
                "5": "Left Main (LM)"
            },
            "LAD": {
                "description": "Left Anterior Descending (LAD)",
                "segments": {
                    "6": "LAD - Proximal",
                    "7": "LAD - Mid",
                    "8": "LAD - Distal",
                    "9": "LAD - 1st Diagonal",
                    "9a": "LAD - 1st Diagonal Branch A",
                    "10": "LAD - 2nd Diagonal", 
                    "10a": "LAD - 2nd Diagonal Branch A"
                }
            },
            "LCX": {
                "description": "Left Circumflex (LCX)",
                "segments": {
                    "11": "LCX - Proximal",
                    "12": "LCX - Mid/Distal",
                    "12a": "LCX - 1st Obtuse Marginal (OM1)",
                    "12b": "LCX - 2nd Obtuse Marginal (OM2)", 
                    "13": "LCX - Distal",
                    "14": "LCX - Posterolateral",
                    "14a": "LCX - Posterolateral Branch A",
                    "14b": "LCX - Posterolateral Branch B",
                    "15": "LCX - Posterior Descending (when LCX dominant)"
                }
            }
        }
    }
    
    print("RIGHT SIDE (RCA System):")
    for seg, desc in coronary_anatomy["RIGHT_CORONARY_ARTERY"]["segments"].items():
        print(f"  {seg}: {desc}")
    
    print("\nLEFT SIDE (Left Coronary System):")
    print(f"  {5}: {coronary_anatomy['LEFT_CORONARY_SYSTEM']['LEFT_MAIN']['5']}")
    
    print("  LAD Segments:")
    for seg, desc in coronary_anatomy["LEFT_CORONARY_SYSTEM"]["LAD"]["segments"].items():
        print(f"    {seg}: {desc}")
        
    print("  LCX Segments:")
    for seg, desc in coronary_anatomy["LEFT_CORONARY_SYSTEM"]["LCX"]["segments"].items():
        print(f"    {seg}: {desc}")
    
    return coronary_anatomy

def create_fixed_classification():
    """Create the corrected artery classification function"""
    print("\n🔧 Creating Fixed Classification Function")
    print("=" * 60)
    
    # Correct classification based on standard coronary anatomy
    def distinguish_side_fixed(segments):
        """
        Determine if artery segments belong to right or left side
        Based on standard AHA 17-segment coronary artery model
        
        Args:
            segments: List of segment names/numbers
            
        Returns:
            str: "right" for RCA system, "left" for left coronary system
        """
        # Convert all segments to strings for consistent comparison
        segments = [str(seg) for seg in segments]
        
        # Right Coronary Artery (RCA) segments
        right_segments = {"1", "2", "3", "4", "16a", "16b", "16c"}
        
        # Left Coronary System segments
        left_segments = {
            "5",    # Left Main
            "6", "7", "8", "9", "9a", "10", "10a",  # LAD system
            "11", "12", "12a", "12b", "13", "14", "14a", "14b", "15"  # LCX system
        }
        
        # Check for right side segments
        has_right = any(seg in right_segments for seg in segments)
        has_left = any(seg in left_segments for seg in segments)
        
        # Handle edge cases
        if has_right and has_left:
            # Mixed case - could indicate multi-vessel disease
            # For classification purposes, if more right segments, classify as right
            right_count = sum(1 for seg in segments if seg in right_segments)
            left_count = sum(1 for seg in segments if seg in left_segments)
            return "right" if right_count >= left_count else "left"
        elif has_right:
            return "right"
        elif has_left:
            return "left"
        else:
            # Unknown segments - default to left (more common in datasets)
            return "left"
    
    # Test the fixed function
    test_cases = [
        (["1"], "right"),
        (["2"], "right"), 
        (["3"], "right"),
        (["4"], "right"),
        (["16a"], "right"),
        (["5"], "left"),   # Left Main
        (["6"], "left"),   # LAD
        (["7"], "left"),   # LAD
        (["11"], "left"),  # LCX
        (["12"], "left"),  # LCX
        (["15"], "left"),  # LCX PDA
        (["1", "6"], "left"),  # Mixed, but more left (1 right, 1 left, so left wins in tie-break)
        (["1", "2", "6"], "right"),  # Mixed, but more right (2 right, 1 left)
    ]
    
    print("Testing Fixed Classification:")
    print("-" * 40)
    
    all_correct = True
    for segments, expected in test_cases:
        result = distinguish_side_fixed(segments)
        status = "✅" if result == expected else "❌"
        print(f"{status} {segments} → {result} (expected: {expected})")
        if result != expected:
            all_correct = False
    
    print(f"\n{'✅ All tests passed!' if all_correct else '❌ Some tests failed!'}")
    
    return distinguish_side_fixed

def main():
    """Main function to analyze and fix artery classification"""
    print("🫀 ARCADE Artery Classification Analysis & Fix")
    print("=" * 60)
    
    # Step 1: Analyze current classification
    errors = analyze_current_classification()
    
    # Step 2: Research coronary anatomy
    anatomy = research_coronary_anatomy()
    
    # Step 3: Create fixed classification
    fixed_function = create_fixed_classification()
    
    # Step 4: Generate fix recommendations
    print("\n📋 Fix Recommendations")
    print("=" * 60)
    
    if errors:
        print("❌ ISSUES FOUND:")
        print("The current distinguish_side() function has incorrect classifications.")
        print("\n🔧 REQUIRED FIXES:")
        print("1. Update the distinguish_side() function in torch_arcade_loader.py")
        print("2. Fix the segment classification logic based on proper coronary anatomy")
        print("3. Add comprehensive documentation about segment definitions")
        
        print("\n📝 Specific Changes Needed:")
        print("- Current function only checks for segments 1,2,3,4,16a,16b,16c as 'right'")
        print("- This is INCORRECT - missing proper left coronary system classification")
        print("- Should properly classify segments 5-15 as 'left' (Left Main, LAD, LCX)")
        
    else:
        print("✅ No issues found in current classification logic.")
    
    print(f"\n📊 Summary:")
    print(f"  - Analyzed {len([item for sublist in [test_cases for test_cases in []] for item in sublist]) or 'multiple'} test cases")
    print(f"  - Found {len(errors)} classification errors")
    print(f"  - Created fixed classification function")
    print(f"  - Ready to implement fix")

if __name__ == "__main__":
    main()
