#!/usr/bin/env python3
"""AGAN AI Agent Pipeline - CLI Demo"""
import sys
sys.path.insert(0, '/tmp')
from agan_pipeline import make_pipeline, PrivacyFirewall
from agan_schemas import uuid4_str

def main():
    pipeline = make_pipeline()
    pf = PrivacyFirewall()
    
    print("=" * 60)
    print("AGAN AI Agent Pipeline - CLI Demo")
    print("=" * 60)
    print("Type 'quit' to exit, 'canary' to run privacy canary tests")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if query.lower() == 'canary':
            passed = pf.verify_canary()
            print(f"Privacy canary tests: {'PASSED' if passed else 'FAILED'}")
            continue
        
        # Privacy scan
        scanned, flags = pf.scan_query(query)
        if flags:
            print(f"[Privacy flags: {', '.join(flags)}]")
        
        # Run pipeline
        request = {
            "query": scanned,
            "user_id": "cli_user",
            "hardware": "M",
            "model_size_b": 7.0,
            "context": {},
        }
        output = pipeline["orchestrator"].run(request)
        answer = output.get("answer")
        
        if answer:
            print(f"\n{answer.text}")
            print(f"\n[Confidence: {answer.confidence:.2f} | Route: {output.get('route', {}).type}]")
        else:
            print("No answer generated.")

if __name__ == "__main__":
    main()
