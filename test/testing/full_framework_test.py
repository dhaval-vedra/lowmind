"""
LowMind Full Framework Sanity Check Test
Executes and validates all demo and test files to ensure full framework compatibility.
"""
import subprocess
import os
import sys

# All target test and demo scripts to run
SCRIPTS_TO_RUN = [
    "test/Advanced Model with Training Visualization.py",
    "test/Guaranteed Output.py",
    "test/IoT Sensor Analyzer.py",
    "test/MNIST Digit Classifier.py",
    "test/MNIST डिजिट क्लासिफिकेशन.py",
    "test/MNIST.py",
    "test/Performance Benchmark.py",
    "test/Smart Agriculture System.py",
    "test/all feature.py",
    "test/home otmation.py",
    "test/testing/API Consistency Testing.py",
    "test/testing/Comprehensive Unit Tests.py",
    "test/testing/Documentation Testing.py",
    "test/testing/Edge Cases Testing.py",
    "test/testing/Performance Benchmarking.py",
    "test/testing/Advanced Features Tests.py"
]

def run_sanity_checks():
    print("=" * 60)
    print("🚀 STARTING LOWMIND FULL FRAMEWORK SANITY CHECK")
    print("=" * 60)

    passed_count = 0
    failed_scripts = []

    # Set PYTHONPATH to the root directory
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    for script in SCRIPTS_TO_RUN:
        print(f"\n🏃 Running: {script} ...")
        try:
            # Run script as subprocess and hide verbose output unless it fails
            res = subprocess.run(
                [sys.executable, script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            if res.returncode == 0:
                print(f"✅ PASSED: {script}")
                passed_count += 1
            else:
                print(f"❌ FAILED: {script}")
                print("--- Error Output ---")
                print(res.stderr)
                failed_scripts.append((script, res.stderr))
        except Exception as e:
            print(f"❌ EXCEPTION running {script}: {e}")
            failed_scripts.append((script, str(e)))

    print("\n" + "=" * 60)
    print("📊 SANITY CHECK RESULTS")
    print("=" * 60)
    print(f"Passed: {passed_count}/{len(SCRIPTS_TO_RUN)}")

    if failed_scripts:
        print(f"Failed: {len(failed_scripts)}")
        for script, err in failed_scripts:
            print(f"  - {script}")
        sys.exit(1)
    else:
        print("🎉 ALL COMPONENT PARTS WORKING EXCELLENTLY AND IN HARMONY!")
        sys.exit(0)

if __name__ == "__main__":
    run_sanity_checks()
