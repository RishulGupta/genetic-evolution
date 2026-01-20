import requests
import sys
import json
from datetime import datetime

class CodebaseEvolutionGenomeAPITester:
    def __init__(self, base_url="https://codehealth-18.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test_name": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {name}")
        if details:
            print(f"   Details: {details}")

    def test_root_endpoint(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Message: {data.get('message', 'N/A')}, Version: {data.get('version', 'N/A')}"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("API Root Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("API Root Endpoint", False, f"Error: {str(e)}")
            return False

    def test_rate_limit_endpoint(self):
        """Test GitHub API rate limit endpoint"""
        try:
            response = requests.get(f"{self.api_url}/rate-limit", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Limit: {data.get('limit', 'N/A')}, Remaining: {data.get('remaining', 'N/A')}"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("Rate Limit Endpoint", success, details)
            return success, data if success else {}
        except Exception as e:
            self.log_test("Rate Limit Endpoint", False, f"Error: {str(e)}")
            return False, {}

    def test_analyze_repo_endpoint(self, repo_url="octocat/Hello-World"):
        """Test repository analysis endpoint"""
        try:
            payload = {
                "url": repo_url,
                "force_refresh": False
            }
            
            print(f"   Analyzing repository: {repo_url}")
            response = requests.post(
                f"{self.api_url}/analyze-repo", 
                json=payload, 
                timeout=30
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                health_score = data.get('health_score', {})
                overall_score = health_score.get('overall_score', 0)
                risk_level = health_score.get('risk_level', 'Unknown')
                total_commits = data.get('analysis_summary', {}).get('total_commits', 0)
                
                details = f"Health Score: {overall_score}, Risk: {risk_level}, Commits: {total_commits}"
            else:
                try:
                    error_data = response.json()
                    details = f"Status: {response.status_code}, Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details = f"Status: {response.status_code}, Response: {response.text[:200]}"
            
            self.log_test("Repository Analysis", success, details)
            return success, data if success else {}
        except Exception as e:
            self.log_test("Repository Analysis", False, f"Error: {str(e)}")
            return False, {}

    def test_compare_repos_endpoint(self, repo_a="octocat/Hello-World", repo_b="octocat/Spoon-Knife"):
        """Test repository comparison endpoint"""
        try:
            payload = {
                "repo_a_url": repo_a,
                "repo_b_url": repo_b
            }
            
            print(f"   Comparing repositories: {repo_a} vs {repo_b}")
            response = requests.post(
                f"{self.api_url}/compare-repos", 
                json=payload, 
                timeout=60
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                comparison = data.get('comparison', {})
                winner = comparison.get('winner', 'Unknown')
                explanation = comparison.get('explanation', '')[:100] + "..."
                
                details = f"Winner: {winner}, Explanation: {explanation}"
            else:
                try:
                    error_data = response.json()
                    details = f"Status: {response.status_code}, Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details = f"Status: {response.status_code}, Response: {response.text[:200]}"
            
            self.log_test("Repository Comparison", success, details)
            return success, data if success else {}
        except Exception as e:
            self.log_test("Repository Comparison", False, f"Error: {str(e)}")
            return False, {}

    def test_predict_bug_endpoint(self):
        """Test bug prediction endpoint"""
        try:
            # Sample DNA features for testing
            payload = {
                "commit_size": 150,
                "files_touched": 5,
                "time_gap_hours": 24.0,
                "code_churn_score": 0.3,
                "commit_type": "feature",
                "is_bug_fix": False
            }
            
            response = requests.post(
                f"{self.api_url}/predict-bug", 
                json=payload, 
                timeout=10
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                is_bug_prone = data.get('is_bug_prone', False)
                confidence = data.get('confidence', 0)
                
                details = f"Bug Prone: {is_bug_prone}, Confidence: {confidence:.2f}"
            else:
                try:
                    error_data = response.json()
                    details = f"Status: {response.status_code}, Error: {error_data.get('detail', 'Unknown error')}"
                except:
                    details = f"Status: {response.status_code}"
            
            self.log_test("Bug Prediction", success, details)
            return success
        except Exception as e:
            self.log_test("Bug Prediction", False, f"Error: {str(e)}")
            return False

    def test_health_score_endpoint(self, owner="octocat", repo="Hello-World"):
        """Test cached health score endpoint"""
        try:
            response = requests.get(
                f"{self.api_url}/health-score/{owner}/{repo}", 
                timeout=10
            )
            
            # This might return 404 if not cached, which is acceptable
            success = response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                overall_score = data.get('overall_score', 0)
                details = f"Cached Health Score: {overall_score}"
            elif response.status_code == 404:
                details = "Repository not analyzed yet (expected for fresh test)"
            else:
                details = f"Status: {response.status_code}"
            
            self.log_test("Health Score Retrieval", success, details)
            return success
        except Exception as e:
            self.log_test("Health Score Retrieval", False, f"Error: {str(e)}")
            return False

    def run_comprehensive_test(self):
        """Run all API tests"""
        print("🧬 Starting Codebase Evolution Genome API Tests")
        print("=" * 60)
        
        # Test basic connectivity
        if not self.test_root_endpoint():
            print("❌ API is not accessible. Stopping tests.")
            return False
        
        # Test rate limit endpoint
        rate_limit_success, rate_limit_data = self.test_rate_limit_endpoint()
        
        # Check if we have enough API calls remaining
        if rate_limit_success:
            remaining = rate_limit_data.get('remaining', 0)
            if remaining < 10:
                print(f"⚠️  Warning: Only {remaining} GitHub API calls remaining")
        
        # Test repository analysis (core functionality)
        analysis_success, analysis_data = self.test_analyze_repo_endpoint()
        
        # Test repository comparison (if analysis worked)
        if analysis_success:
            self.test_compare_repos_endpoint()
        else:
            print("⚠️  Skipping comparison test due to analysis failure")
        
        # Test bug prediction
        self.test_predict_bug_endpoint()
        
        # Test health score retrieval
        self.test_health_score_endpoint()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("✅ Backend API is functioning well!")
        elif success_rate >= 60:
            print("⚠️  Backend API has some issues but core functionality works")
        else:
            print("❌ Backend API has significant issues")
        
        return success_rate >= 60

def main():
    """Main test execution"""
    tester = CodebaseEvolutionGenomeAPITester()
    
    try:
        success = tester.run_comprehensive_test()
        
        # Save detailed results
        with open('/app/backend_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'tests_run': tester.tests_run,
                    'tests_passed': tester.tests_passed,
                    'success_rate': (tester.tests_passed / tester.tests_run) * 100 if tester.tests_run > 0 else 0,
                    'timestamp': datetime.now().isoformat()
                },
                'detailed_results': tester.test_results
            }, f, indent=2)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())