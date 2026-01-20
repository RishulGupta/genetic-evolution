from fastapi import FastAPI, APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import asyncio
import re
from collections import Counter
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# GitHub API Configuration
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')  # Optional for rate-limited access

# Create the main app
app = FastAPI(title="Codebase Evolution Genome API", version="1.0.0")

# Create API router
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= MODELS =============

class CommitData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sha: str
    message: str
    author_name: str
    author_email: str
    author_date: str
    url: str
    additions: int = 0
    deletions: int = 0
    files_changed: int = 0
    is_bug_fix: bool = False
    commit_type: str = "feature"


class ContributorData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    login: str
    contributions: int
    avatar_url: str
    profile_url: str


class ForkData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    owner: str
    url: str
    created_at: str
    stargazers_count: int


class RepositoryData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    owner: str
    repo: str
    full_name: str
    description: Optional[str] = None
    stargazers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    created_at: str
    updated_at: str
    language: Optional[str] = None
    last_analyzed: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DNAFeatures(BaseModel):
    model_config = ConfigDict(extra="ignore")
    commit_size: int  # additions + deletions
    files_touched: int
    time_gap_hours: float  # hours since previous commit
    code_churn_score: float  # ratio of deletions to additions
    commit_type: str  # bug/feature/refactor
    is_bug_fix: bool


class HealthScore(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repository_id: str
    owner: str
    repo: str
    overall_score: float  # 0-100
    bug_frequency_score: float
    commit_stability_score: float
    contributor_diversity_score: float
    change_volatility_score: float
    risk_level: str  # Low/Medium/High
    calculated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repo_a: str
    repo_b: str
    winner: str
    explanation: str
    metrics: Dict[str, Any]
    compared_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BugPrediction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    commit_sha: str
    is_bug_prone: bool
    confidence: float
    features: Dict[str, Any]


class RepositoryAnalysisRequest(BaseModel):
    url: str  # GitHub repo URL
    force_refresh: bool = False


class CompareRepositoriesRequest(BaseModel):
    repo_a_url: str
    repo_b_url: str


# ============= GITHUB SERVICE =============

class GitHubService:
    def __init__(self):
        self.base_url = GITHUB_API_URL
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        if GITHUB_TOKEN:
            self.headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    
    async def check_rate_limit(self) -> Dict:
        """Check current rate limit status"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/rate_limit",
                    headers=self.headers,
                    timeout=10.0
                )
                return response.json()
            except Exception as e:
                logger.error(f"Error checking rate limit: {e}")
                return {"resources": {"core": {"remaining": 0}}}
    
    def parse_repo_url(self, url: str) -> tuple:
        """Extract owner and repo name from GitHub URL"""
        # Handle formats: https://github.com/owner/repo or owner/repo
        pattern = r'github\.com/([^/]+)/([^/\s]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2).replace('.git', '')
        
        # Try simple format
        parts = url.strip().split('/')
        if len(parts) >= 2:
            return parts[-2], parts[-1].replace('.git', '')
        
        raise ValueError("Invalid GitHub repository URL")
    
    async def fetch_repository_info(self, owner: str, repo: str) -> RepositoryData:
        """Fetch repository metadata"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/repos/{owner}/{repo}",
                    headers=self.headers,
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                
                return RepositoryData(
                    owner=owner,
                    repo=repo,
                    full_name=data['full_name'],
                    description=data.get('description'),
                    stargazers_count=data['stargazers_count'],
                    forks_count=data['forks_count'],
                    open_issues_count=data['open_issues_count'],
                    created_at=data['created_at'],
                    updated_at=data['updated_at'],
                    language=data.get('language')
                )
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"GitHub API error: {e}")
    
    async def fetch_commits(self, owner: str, repo: str, max_pages: int = 3) -> List[CommitData]:
        """Fetch commits with detailed information"""
        commits = []
        
        async with httpx.AsyncClient() as client:
            for page in range(1, max_pages + 1):
                try:
                    response = await client.get(
                        f"{self.base_url}/repos/{owner}/{repo}/commits",
                        headers=self.headers,
                        params={"per_page": 30, "page": page},
                        timeout=15.0
                    )
                    
                    if response.status_code == 429:
                        logger.warning("Rate limit hit, stopping fetch")
                        break
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        break
                    
                    for commit in data:
                        # Detect bug-related commits
                        message = commit['commit']['message'].lower()
                        is_bug = any(keyword in message for keyword in ['fix', 'bug', 'error', 'crash', 'hotfix', 'patch'])
                        commit_type = 'bug' if is_bug else ('refactor' if 'refactor' in message else 'feature')
                        
                        commit_obj = CommitData(
                            sha=commit['sha'],
                            message=commit['commit']['message'],
                            author_name=commit['commit']['author']['name'],
                            author_email=commit['commit']['author']['email'],
                            author_date=commit['commit']['author']['date'],
                            url=commit['html_url'],
                            is_bug_fix=is_bug,
                            commit_type=commit_type
                        )
                        
                        # Fetch detailed commit info for stats
                        try:
                            detail_response = await client.get(
                                f"{self.base_url}/repos/{owner}/{repo}/commits/{commit['sha']}",
                                headers=self.headers,
                                timeout=10.0
                            )
                            if detail_response.status_code == 200:
                                detail = detail_response.json()
                                stats = detail.get('stats', {})
                                commit_obj.additions = stats.get('additions', 0)
                                commit_obj.deletions = stats.get('deletions', 0)
                                commit_obj.files_changed = len(detail.get('files', []))
                        except:
                            pass
                        
                        commits.append(commit_obj)
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error fetching commits page {page}: {e}")
                    break
        
        return commits
    
    async def fetch_contributors(self, owner: str, repo: str) -> List[ContributorData]:
        """Fetch repository contributors"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/repos/{owner}/{repo}/contributors",
                    headers=self.headers,
                    params={"per_page": 100},
                    timeout=10.0
                )
                response.raise_for_status()
                
                contributors = []
                for contrib in response.json():
                    contributors.append(ContributorData(
                        login=contrib['login'],
                        contributions=contrib['contributions'],
                        avatar_url=contrib['avatar_url'],
                        profile_url=contrib['html_url']
                    ))
                
                return contributors
            except Exception as e:
                logger.error(f"Error fetching contributors: {e}")
                return []
    
    async def fetch_forks(self, owner: str, repo: str, max_pages: int = 2) -> List[ForkData]:
        """Fetch repository forks"""
        forks = []
        
        async with httpx.AsyncClient() as client:
            for page in range(1, max_pages + 1):
                try:
                    response = await client.get(
                        f"{self.base_url}/repos/{owner}/{repo}/forks",
                        headers=self.headers,
                        params={"per_page": 30, "page": page, "sort": "newest"},
                        timeout=10.0
                    )
                    
                    if response.status_code == 429:
                        break
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        break
                    
                    for fork in data:
                        forks.append(ForkData(
                            name=fork['name'],
                            owner=fork['owner']['login'],
                            url=fork['html_url'],
                            created_at=fork['created_at'],
                            stargazers_count=fork['stargazers_count']
                        ))
                    
                except Exception as e:
                    logger.error(f"Error fetching forks: {e}")
                    break
        
        return forks


github_service = GitHubService()


# ============= ANALYSIS ENGINE =============

class AnalysisEngine:
    """DNA Feature Engineering and Health Scoring"""
    
    def calculate_dna_features(self, commits: List[CommitData]) -> List[DNAFeatures]:
        """Calculate DNA features for each commit"""
        features = []
        
        for i, commit in enumerate(commits):
            commit_size = commit.additions + commit.deletions
            files_touched = commit.files_changed
            
            # Calculate time gap from previous commit
            time_gap_hours = 0.0
            if i < len(commits) - 1:
                try:
                    current_time = datetime.fromisoformat(commit.author_date.replace('Z', '+00:00'))
                    previous_time = datetime.fromisoformat(commits[i + 1].author_date.replace('Z', '+00:00'))
                    time_gap_hours = abs((current_time - previous_time).total_seconds() / 3600)
                except:
                    time_gap_hours = 24.0
            
            # Code churn score
            churn_score = 0.0
            if commit.additions > 0:
                churn_score = commit.deletions / commit.additions
            
            features.append(DNAFeatures(
                commit_size=commit_size,
                files_touched=files_touched,
                time_gap_hours=time_gap_hours,
                code_churn_score=churn_score,
                commit_type=commit.commit_type,
                is_bug_fix=commit.is_bug_fix
            ))
        
        return features
    
    def calculate_health_score(
        self, 
        repo_data: RepositoryData,
        commits: List[CommitData],
        contributors: List[ContributorData],
        forks: List[ForkData]
    ) -> HealthScore:
        """Calculate comprehensive health score (0-100)"""
        
        # 1. Bug Frequency Score (25 points)
        total_commits = len(commits)
        bug_commits = sum(1 for c in commits if c.is_bug_fix)
        bug_ratio = bug_commits / total_commits if total_commits > 0 else 0
        bug_frequency_score = max(0, 25 - (bug_ratio * 100))  # Lower bugs = higher score
        
        # 2. Commit Stability Score (25 points)
        if total_commits > 1:
            commit_sizes = [c.additions + c.deletions for c in commits]
            avg_size = np.mean(commit_sizes)
            std_size = np.std(commit_sizes)
            cv = (std_size / avg_size) if avg_size > 0 else 1.0  # Coefficient of variation
            commit_stability_score = max(0, 25 - (cv * 10))  # Lower variance = higher stability
        else:
            commit_stability_score = 15.0
        
        # 3. Contributor Diversity Score (25 points)
        num_contributors = len(contributors)
        if num_contributors > 0:
            contributions = [c.contributions for c in contributors]
            total_contributions = sum(contributions)
            # Gini coefficient for contribution distribution
            sorted_contrib = sorted(contributions)
            n = len(sorted_contrib)
            gini = (2 * sum((i + 1) * x for i, x in enumerate(sorted_contrib))) / (n * total_contributions) - (n + 1) / n
            contributor_diversity_score = 25 * (1 - gini)  # Lower Gini = better diversity
        else:
            contributor_diversity_score = 5.0
        
        # 4. Change Volatility Score (25 points)
        if total_commits > 5:
            recent_commits = commits[:10]  # Last 10 commits
            recent_bug_ratio = sum(1 for c in recent_commits if c.is_bug_fix) / len(recent_commits)
            change_volatility_score = max(0, 25 - (recent_bug_ratio * 50))
        else:
            change_volatility_score = 15.0
        
        # Overall score
        overall_score = (
            bug_frequency_score +
            commit_stability_score +
            contributor_diversity_score +
            change_volatility_score
        )
        
        # Risk level
        if overall_score >= 70:
            risk_level = "Low"
        elif overall_score >= 40:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return HealthScore(
            repository_id=repo_data.id,
            owner=repo_data.owner,
            repo=repo_data.repo,
            overall_score=round(overall_score, 2),
            bug_frequency_score=round(bug_frequency_score, 2),
            commit_stability_score=round(commit_stability_score, 2),
            contributor_diversity_score=round(contributor_diversity_score, 2),
            change_volatility_score=round(change_volatility_score, 2),
            risk_level=risk_level
        )
    
    def compare_projects(
        self,
        repo_a: RepositoryData,
        health_a: HealthScore,
        commits_a: List[CommitData],
        repo_b: RepositoryData,
        health_b: HealthScore,
        commits_b: List[CommitData]
    ) -> ComparisonResult:
        """Compare two projects and determine the healthier one"""
        
        metrics = {
            "repo_a": {
                "health_score": health_a.overall_score,
                "bug_ratio": sum(1 for c in commits_a if c.is_bug_fix) / len(commits_a) if commits_a else 0,
                "total_commits": len(commits_a),
                "risk_level": health_a.risk_level
            },
            "repo_b": {
                "health_score": health_b.overall_score,
                "bug_ratio": sum(1 for c in commits_b if c.is_bug_fix) / len(commits_b) if commits_b else 0,
                "total_commits": len(commits_b),
                "risk_level": health_b.risk_level
            }
        }
        
        # Determine winner
        winner = repo_a.full_name if health_a.overall_score > health_b.overall_score else repo_b.full_name
        winner_health = health_a if health_a.overall_score > health_b.overall_score else health_b
        loser_health = health_b if health_a.overall_score > health_b.overall_score else health_a
        winner_metrics = metrics["repo_a"] if winner == repo_a.full_name else metrics["repo_b"]
        loser_metrics = metrics["repo_b"] if winner == repo_a.full_name else metrics["repo_a"]
        
        # Generate explanation
        score_diff = abs(health_a.overall_score - health_b.overall_score)
        bug_comparison = "fewer bugs" if winner_metrics["bug_ratio"] < loser_metrics["bug_ratio"] else "similar bug rates"
        
        explanation = (
            f"{winner} is healthier with a score of {winner_health.overall_score:.1f}/100 "
            f"compared to {loser_health.overall_score:.1f}/100. "
            f"It has {bug_comparison}, "
            f"{'stable' if winner_health.commit_stability_score > 18 else 'moderate'} update patterns, "
            f"and {winner_health.risk_level.lower()} risk level. "
            f"The {score_diff:.1f} point advantage suggests "
            f"{'significantly' if score_diff > 20 else 'moderately'} better long-term maintainability."
        )
        
        return ComparisonResult(
            repo_a=repo_a.full_name,
            repo_b=repo_b.full_name,
            winner=winner,
            explanation=explanation,
            metrics=metrics
        )


analysis_engine = AnalysisEngine()


# ============= ML MODEL =============

class BugPredictor:
    """Machine Learning model for bug prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train_model(self, commits: List[CommitData], dna_features: List[DNAFeatures]):
        """Train Random Forest model on commit data"""
        if len(commits) < 10:
            return  # Not enough data
        
        # Prepare features
        X = []
        y = []
        
        for commit, features in zip(commits, dna_features):
            X.append([
                features.commit_size,
                features.files_touched,
                features.time_gap_hours,
                features.code_churn_score
            ])
            y.append(1 if commit.is_bug_fix else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Model trained on {len(commits)} commits")
    
    def predict(self, dna_features: DNAFeatures) -> BugPrediction:
        """Predict if a commit is bug-prone"""
        if not self.is_trained or self.model is None:
            return BugPrediction(
                commit_sha="unknown",
                is_bug_prone=False,
                confidence=0.0,
                features={}
            )
        
        X = np.array([[
            dna_features.commit_size,
            dna_features.files_touched,
            dna_features.time_gap_hours,
            dna_features.code_churn_score
        ]])
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        confidence = np.max(self.model.predict_proba(X_scaled))
        
        return BugPrediction(
            commit_sha="predicted",
            is_bug_prone=bool(prediction),
            confidence=float(confidence),
            features=dna_features.model_dump()
        )


bug_predictor = BugPredictor()


# ============= API ROUTES =============

@api_router.get("/")
async def root():
    return {"message": "Codebase Evolution Genome API", "version": "1.0.0"}


@api_router.get("/rate-limit")
async def get_rate_limit():
    """Get GitHub API rate limit status"""
    rate_limit = await github_service.check_rate_limit()
    core = rate_limit.get('resources', {}).get('core', {})
    return {
        "limit": core.get('limit', 60),
        "remaining": core.get('remaining', 0),
        "reset": core.get('reset', 0)
    }


@api_router.post("/analyze-repo")
async def analyze_repository(request: RepositoryAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze a GitHub repository"""
    try:
        owner, repo = github_service.parse_repo_url(request.url)
        
        # Check cache
        if not request.force_refresh:
            cached = await db.repositories.find_one(
                {"owner": owner, "repo": repo},
                {"_id": 0}
            )
            if cached:
                # Check if last_analyzed exists and is recent
                if 'last_analyzed' in cached:
                    last_analyzed = datetime.fromisoformat(cached['last_analyzed'])
                    if datetime.now(timezone.utc) - last_analyzed < timedelta(hours=1):
                        return cached
                # If no last_analyzed field, treat as old cache and refresh
        
        # Fetch data
        repo_data = await github_service.fetch_repository_info(owner, repo)
        commits, contributors, forks = await asyncio.gather(
            github_service.fetch_commits(owner, repo),
            github_service.fetch_contributors(owner, repo),
            github_service.fetch_forks(owner, repo)
        )
        
        # Calculate DNA features
        dna_features = analysis_engine.calculate_dna_features(commits)
        
        # Calculate health score
        health_score = analysis_engine.calculate_health_score(
            repo_data, commits, contributors, forks
        )
        
        # Train ML model in background
        if len(commits) >= 10:
            background_tasks.add_task(bug_predictor.train_model, commits, dna_features)
        
        # Prepare response
        result = {
            "repository": repo_data.model_dump(),
            "commits": [c.model_dump() for c in commits],
            "contributors": [c.model_dump() for c in contributors],
            "forks": [f.model_dump() for f in forks],
            "dna_features": [f.model_dump() for f in dna_features],
            "health_score": health_score.model_dump(),
            "analysis_summary": {
                "total_commits": len(commits),
                "bug_fixes": sum(1 for c in commits if c.is_bug_fix),
                "total_contributors": len(contributors),
                "total_forks": len(forks)
            }
        }
        
        # Cache result
        await db.repositories.update_one(
            {"owner": owner, "repo": repo},
            {"$set": result},
            upsert=True
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing repository: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@api_router.post("/compare-repos")
async def compare_repositories(request: CompareRepositoriesRequest):
    """Compare two GitHub repositories"""
    try:
        # Analyze both repositories
        analysis_a = await analyze_repository(
            RepositoryAnalysisRequest(url=request.repo_a_url, force_refresh=False),
            BackgroundTasks()
        )
        analysis_b = await analyze_repository(
            RepositoryAnalysisRequest(url=request.repo_b_url, force_refresh=False),
            BackgroundTasks()
        )
        
        # Parse repository data
        repo_a = RepositoryData(**analysis_a['repository'])
        repo_b = RepositoryData(**analysis_b['repository'])
        
        health_a = HealthScore(**analysis_a['health_score'])
        health_b = HealthScore(**analysis_b['health_score'])
        
        commits_a = [CommitData(**c) for c in analysis_a['commits']]
        commits_b = [CommitData(**c) for c in analysis_b['commits']]
        
        # Compare
        comparison = analysis_engine.compare_projects(
            repo_a, health_a, commits_a,
            repo_b, health_b, commits_b
        )
        
        result = {
            "comparison": comparison.model_dump(),
            "repo_a_analysis": analysis_a,
            "repo_b_analysis": analysis_b
        }
        
        # Store comparison
        await db.comparisons.insert_one(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error comparing repositories: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@api_router.post("/predict-bug")
async def predict_bug(dna_features: DNAFeatures):
    """Predict if a commit is bug-prone"""
    try:
        prediction = bug_predictor.predict(dna_features)
        return prediction.model_dump()
    except Exception as e:
        logger.error(f"Error predicting bug: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@api_router.get("/health-score/{owner}/{repo}")
async def get_health_score(owner: str, repo: str):
    """Get cached health score for a repository"""
    cached = await db.repositories.find_one(
        {"owner": owner, "repo": repo},
        {"_id": 0, "health_score": 1}
    )
    
    if not cached:
        raise HTTPException(status_code=404, detail="Repository not analyzed yet")
    
    return cached.get('health_score', {})


# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
