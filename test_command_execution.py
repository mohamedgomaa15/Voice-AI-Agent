"""
Test command execution system
"""

from command_executer import CommandExecutor
from extract_entities import MultilingualHybridSystem
import time

def test_execution_pipeline():
    """Test the complete pipeline: intent → entities → execution"""
    
    executor = CommandExecutor()
    system = MultilingualHybridSystem()
    
    # Test commands
    test_commands = [
        # English commands
        ("open youtube", "Should open YouTube"),
        ("search for machine learning", "Should search Google"),
        ("open youtube and search for AI", "Should open YouTube with search"),
        ("turn up the volume", "Should increase volume"),
        
        # Arabic commands
        ("افتح يوتيوب", "Should open YouTube"),
        ("ابحث عن ذكاء اصطناعي", "Should search for AI"),
        ("افتح يوتيوب وابحث عن موسيقى", "Should open YouTube and search"),
        ("زود الصوت", "Should increase volume"),
    ]
    
    print("="*70)
    print("COMMAND EXECUTION TEST")
    print("="*70)
    
    for i, (command, description) in enumerate(test_commands, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_commands)}: {description}")
        print(f"{'='*70}")
        print(f"Command: {command}")
        
        # Step 1: Process query
        result = system.process_query_optimized(command)
        
        print(f"\n📋 Analysis:")
        print(f"  Language:   {result['language']}")
        print(f"  Intent:     {result['intent']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Entities:   {result['entities']}")
        
        # Step 2: Execute command
        print(f"\n⚙️ Execution:")
        success, message = executor.execute_command(
            result['intent'],
            result['entities'],
            result['language']
        )
        
        status = "✅" if success else "❌"
        print(f"  {status} {message}")
        
        # Wait before next test
        if i < len(test_commands):
            print(f"\n⏳ Waiting 2 seconds before next test...")
            time.sleep(2)
    
    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETED")
    print("="*70)
    print("\nNote: Some commands (like volume control) may require")
    print("additional libraries (pycaw for Windows)")

def interactive_test():
    """Interactive testing mode"""
    executor = CommandExecutor()
    system = MultilingualHybridSystem()
    
    print("="*70)
    print("INTERACTIVE COMMAND EXECUTION")
    print("="*70)
    print("Enter commands to test (or 'exit' to quit)")
    print("Examples:")
    print("  - open youtube")
    print("  - افتح يوتيوب")
    print("  - search for cats")
    print("  - ابحث عن قطط")
    print("="*70)
    
    while True:
        try:
            command = input("\n💬 Command: ").strip()
            
            if command.lower() == 'exit':
                break
            
            if not command:
                continue
            
            # Process
            result = system.process_query_optimized(command)
            
            print(f"\n📋 Analysis:")
            print(f"  Language:   {result['language']}")
            print(f"  Intent:     {result['intent']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  Entities:   {result['entities']}")
            
            # Ask for confirmation
            execute = input(f"\n❓ Execute this command? (y/n): ").lower()
            
            if execute == 'y':
                success, message = executor.execute_command(
                    result['intent'],
                    result['entities'],
                    result['language']
                )
                
                status = "✅" if success else "❌"
                print(f"\n{status} {message}")
            else:
                print("⏭️ Skipped execution")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")

def test_specific_features():
    """Test specific features"""
    executor = CommandExecutor()
    
    print("="*70)
    print("FEATURE-SPECIFIC TESTS")
    print("="*70)
    
    # Test 1: App name normalization
    print("\n1️⃣ Testing App Name Normalization")
    test_names = [
        'YouTube', 'youtube', 'you tube', 'yt',
        'يوتيوب', 'يوتوب',
        'Netflix', 'net flix', 'نتفليكس'
    ]
    
    for name in test_names:
        normalized = executor.normalize_app_name(name)
        print(f"  '{name}' → '{normalized}'")
    
    # Test 2: URL generation
    print("\n2️⃣ Testing Search URL Generation")
    test_queries = [
        ('youtube', 'cats'),
        ('google', 'machine learning'),
        ('netflix', 'action movies'),
    ]
    
    for platform, query in test_queries:
        template = executor.search_urls.get(platform, '')
        if template:
            import urllib.parse
            url = template.format(urllib.parse.quote(query))
            print(f"  {platform} + '{query}':")
            print(f"    {url}")
    
    # Test 3: Platform detection
    print(f"\n3️⃣ Platform Information")
    print(f"  OS: {executor.os_type}")
    print(f"  Supported apps: {len(executor.app_commands.get(executor.os_type, {}))}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == '--interactive':
            interactive_test()
        elif mode == '--features':
            test_specific_features()
        elif mode == '--full':
            test_specific_features()
            print("\n" + "="*70 + "\n")
            test_execution_pipeline()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python test_command_execution.py [--interactive|--features|--full]")
    else:
        # Default: run pipeline test
        test_execution_pipeline()
        
        print("\n💡 Tip: Run with --interactive for interactive testing")
        print("        Run with --features to test specific features")
        print("        Run with --full for comprehensive testing")