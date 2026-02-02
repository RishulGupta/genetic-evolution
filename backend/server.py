from fastapi import FastAPI, APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Set, Tuple
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import asyncio
import re
from collections import Counter, defaultdict
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

class FileChange(BaseModel):
    """Represents a file changed in a commit"""
    model_config = ConfigDict(extra="ignore")
    filename: str
    status: str = "modified"  # added, removed, modified, renamed
    additions: int = 0
    deletions: int = 0
    patch: Optional[str] = None


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
    files: List[FileChange] = []
    is_bug_fix: bool = False
    is_bug_introducing: bool = False
    commit_type: str = "feature"
    parent_shas: List[str] = []


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


# ============= BUG EVOLUTION MODELS =============

class BugLineage(BaseModel):
    """Tracks the lifecycle of a bug from introduction to fix"""
    model_config = ConfigDict(extra="ignore")
    bug_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    introducing_commit: str  # SHA of commit that introduced the bug
    fix_commits: List[str] = []  # SHAs of commits that attempted to fix
    reintroduction_commits: List[str] = []  # SHAs where bug reappeared
    affected_files: List[str] = []
    lifespan_commits: int = 0  # Number of commits bug survived
    lifespan_days: float = 0.0
    fix_attempts: int = 0
    is_resolved: bool = False
    resolution_confidence: float = 0.0  # 0-1, higher = more confident it's fixed
    first_seen: str = ""
    last_seen: str = ""


class BugPropagation(BaseModel):
    """Tracks how bugs spread through the codebase"""
    model_config = ConfigDict(extra="ignore")
    source_file: str
    target_files: List[str] = []
    propagation_depth: int = 0  # How many files deep it spread
    propagation_width: int = 0  # How many files at same level
    spread_via: str = "file_overlap"  # file_overlap, merge, contributor
    contributor_spread: List[str] = []  # Contributors who touched infected files


class BugContagionScore(BaseModel):
    """Overall contagion metrics for a repository"""
    model_config = ConfigDict(extra="ignore")
    score: float = 0.0  # 0-1 scale
    interpretation: str = "Contained"  # Contained, Moderate, Highly Infectious
    propagation_depth_avg: float = 0.0
    avg_lifespan_days: float = 0.0
    reinfection_rate: float = 0.0
    contributor_spread_factor: float = 0.0


class FileHotspot(BaseModel):
    """Identifies bug-prone files/modules"""
    model_config = ConfigDict(extra="ignore")
    filename: str
    bug_count: int = 0
    bug_fix_count: int = 0
    churn_score: float = 0.0
    hotspot_intensity: float = 0.0  # 0-1 scale
    last_bug_date: str = ""
    contributors_involved: List[str] = []


class SuperSpreaderCommit(BaseModel):
    """Commits that introduced bugs affecting many files"""
    model_config = ConfigDict(extra="ignore")
    sha: str
    message: str
    author: str
    files_affected: int = 0
    bugs_introduced: int = 0
    amplification_score: float = 0.0


class RecoveryMetrics(BaseModel):
    """Measures how well a project recovers from bugs"""
    model_config = ConfigDict(extra="ignore")
    avg_time_to_fix_hours: float = 0.0
    regression_probability: float = 0.0
    clean_commit_streak: int = 0
    immunity_score: float = 0.0  # 0-100
    resilience_class: str = "Unknown"  # Fragile, Moderate, Resilient, Antifragile


class DeveloperInfluence(BaseModel):
    """Systemic analysis of developer patterns (not personal blame)"""
    model_config = ConfigDict(extra="ignore")
    contributor: str
    commits_total: int = 0
    bug_introductions: int = 0
    bug_fixes: int = 0
    regression_associations: int = 0
    net_impact_score: float = 0.0  # Positive = more fixes than introductions


class TemporalBugWave(BaseModel):
    """Detects bug surges after major events"""
    model_config = ConfigDict(extra="ignore")
    wave_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger_event: str = ""  # refactor, release, contributor_churn
    trigger_commit: str = ""
    wave_start: str = ""
    wave_peak: str = ""
    wave_end: str = ""
    bug_count: int = 0
    recovery_duration_days: float = 0.0


class BugEvolutionAnalysis(BaseModel):
    """Complete bug evolution analysis for a repository"""
    model_config = ConfigDict(extra="ignore")
    repository_id: str
    bug_lineages: List[BugLineage] = []
    contagion_score: BugContagionScore = BugContagionScore()
    file_hotspots: List[FileHotspot] = []
    super_spreaders: List[SuperSpreaderCommit] = []
    recovery_metrics: RecoveryMetrics = RecoveryMetrics()
    developer_influence: List[DeveloperInfluence] = []
    temporal_waves: List[TemporalBugWave] = []
    total_bugs_detected: int = 0
    active_bugs: int = 0
    resolved_bugs: int = 0
    analyzed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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
    contagion_penalty: float = 0.0  # Deduction based on bug contagion
    recovery_bonus: float = 0.0  # Bonus for good recovery metrics
    risk_level: str  # Low/Medium/High
    calculated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ComparisonResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repo_a: str
    repo_b: str
    winner: str
    explanation: str
    metrics: Dict[str, Any]
    bug_evolution_comparison: Dict[str, Any] = {}
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
        """Fetch commits with detailed information including file changes"""
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
                        is_bug_fix = any(keyword in message for keyword in ['fix', 'bug', 'error', 'crash', 'hotfix', 'patch', 'resolve', 'issue'])
                        is_bug_introducing = any(keyword in message for keyword in ['revert', 'broke', 'breaking', 'regression'])
                        
                        # Classify commit type
                        if is_bug_fix:
                            commit_type = 'bug'
                        elif 'refactor' in message:
                            commit_type = 'refactor'
                        elif 'chore' in message or 'docs' in message or 'style' in message:
                            commit_type = 'chore'
                        else:
                            commit_type = 'feature'
                        
                        # Get parent SHAs for lineage tracking
                        parent_shas = [p['sha'] for p in commit.get('parents', [])]
                        
                        commit_obj = CommitData(
                            sha=commit['sha'],
                            message=commit['commit']['message'],
                            author_name=commit['commit']['author']['name'],
                            author_email=commit['commit']['author']['email'],
                            author_date=commit['commit']['author']['date'],
                            url=commit['html_url'],
                            is_bug_fix=is_bug_fix,
                            is_bug_introducing=is_bug_introducing,
                            commit_type=commit_type,
                            parent_shas=parent_shas
                        )
                        
                        # Fetch detailed commit info for stats and files
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
                                
                                # Extract file changes for bug tracking
                                files = []
                                for f in detail.get('files', []):
                                    files.append(FileChange(
                                        filename=f.get('filename', ''),
                                        status=f.get('status', 'modified'),
                                        additions=f.get('additions', 0),
                                        deletions=f.get('deletions', 0),
                                        patch=f.get('patch', '')[:500] if f.get('patch') else None  # Truncate patch
                                    ))
                                commit_obj.files = files
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


# ============= BUG EVOLUTION ENGINE =============

class BugEvolutionEngine:
    """Analyzes bug spread, mutation, and evolution patterns"""
    
    def analyze_bug_evolution(
        self,
        commits: List[CommitData],
        contributors: List[ContributorData],
        forks: List[ForkData],
        repo_id: str
    ) -> BugEvolutionAnalysis:
        """Perform comprehensive bug evolution analysis"""
        
        analysis = BugEvolutionAnalysis(repository_id=repo_id)
        
        # Sort commits by date (oldest first for lineage tracking)
        sorted_commits = sorted(
            commits,
            key=lambda c: datetime.fromisoformat(c.author_date.replace('Z', '+00:00'))
        )
        
        # 5.1 Bug Lineage Tracking
        analysis.bug_lineages = self._track_bug_lineages(sorted_commits)
        
        # 5.3 Bug Contagion Score
        analysis.contagion_score = self._calculate_contagion_score(
            analysis.bug_lineages, sorted_commits, contributors
        )
        
        # 5.6 Hotspots & Super-Spreaders
        analysis.file_hotspots = self._identify_hotspots(sorted_commits)
        analysis.super_spreaders = self._identify_super_spreaders(sorted_commits)
        
        # 5.7 Recovery & Immune Response
        analysis.recovery_metrics = self._calculate_recovery_metrics(
            sorted_commits, analysis.bug_lineages
        )
        
        # 5.8 Developer Influence Analysis
        analysis.developer_influence = self._analyze_developer_influence(sorted_commits)
        
        # 5.9 Temporal Bug Waves
        analysis.temporal_waves = self._detect_temporal_waves(sorted_commits)
        
        # Summary stats
        analysis.total_bugs_detected = len(analysis.bug_lineages)
        analysis.active_bugs = sum(1 for b in analysis.bug_lineages if not b.is_resolved)
        analysis.resolved_bugs = sum(1 for b in analysis.bug_lineages if b.is_resolved)
        
        return analysis
    
    def _track_bug_lineages(self, commits: List[CommitData]) -> List[BugLineage]:
        """Track bug lifecycles from introduction to fix"""
        lineages = []
        file_bug_map: Dict[str, List[BugLineage]] = defaultdict(list)
        
        for i, commit in enumerate(commits):
            affected_files = [f.filename for f in commit.files]
            
            if commit.is_bug_fix:
                # This commit fixes bugs - find related lineages
                for filename in affected_files:
                    if filename in file_bug_map:
                        for lineage in file_bug_map[filename]:
                            if not lineage.is_resolved:
                                lineage.fix_commits.append(commit.sha)
                                lineage.fix_attempts += 1
                                lineage.last_seen = commit.author_date
                                
                                # Calculate lifespan
                                try:
                                    first = datetime.fromisoformat(lineage.first_seen.replace('Z', '+00:00'))
                                    last = datetime.fromisoformat(commit.author_date.replace('Z', '+00:00'))
                                    lineage.lifespan_days = (last - first).total_seconds() / 86400
                                except:
                                    pass
                                
                                # Check if truly resolved (no more bug commits on these files after)
                                remaining_commits = commits[i+1:]
                                future_bugs = any(
                                    c.is_bug_fix and filename in [f.filename for f in c.files]
                                    for c in remaining_commits[:10]  # Look ahead 10 commits
                                )
                                
                                if not future_bugs:
                                    lineage.is_resolved = True
                                    lineage.resolution_confidence = 0.8
                                else:
                                    lineage.resolution_confidence = 0.3
            
            elif commit.is_bug_introducing or (commit.commit_type == 'feature' and commit.files_changed > 5):
                # Potential bug-introducing commit
                # Heuristic: Large feature commits or explicit bug-introducing markers
                
                # Check if subsequent commits fix bugs in these files
                subsequent_commits = commits[i+1:i+20]  # Look at next 20 commits
                for filename in affected_files:
                    subsequent_fixes = [
                        c for c in subsequent_commits
                        if c.is_bug_fix and filename in [f.filename for f in c.files]
                    ]
                    
                    if subsequent_fixes:
                        # This file had bugs fixed after this commit
                        lineage = BugLineage(
                            introducing_commit=commit.sha,
                            affected_files=[filename],
                            first_seen=commit.author_date,
                            last_seen=subsequent_fixes[0].author_date if subsequent_fixes else commit.author_date
                        )
                        lineages.append(lineage)
                        file_bug_map[filename].append(lineage)
        
        # Calculate lifespan in commits
        commit_sha_to_idx = {c.sha: i for i, c in enumerate(commits)}
        for lineage in lineages:
            if lineage.introducing_commit in commit_sha_to_idx:
                intro_idx = commit_sha_to_idx[lineage.introducing_commit]
                if lineage.fix_commits:
                    fix_idx = max(
                        commit_sha_to_idx.get(sha, intro_idx)
                        for sha in lineage.fix_commits
                    )
                    lineage.lifespan_commits = fix_idx - intro_idx
        
        return lineages
    
    def _calculate_contagion_score(
        self,
        lineages: List[BugLineage],
        commits: List[CommitData],
        contributors: List[ContributorData]
    ) -> BugContagionScore:
        """Calculate Bug Contagion Score (0-1)"""
        
        if not lineages:
            return BugContagionScore(
                score=0.0,
                interpretation="Contained"
            )
        
        # Calculate component scores
        
        # 1. Average propagation depth (files affected per bug)
        avg_files_affected = np.mean([len(l.affected_files) for l in lineages]) if lineages else 0
        propagation_depth_score = min(avg_files_affected / 10, 1.0)  # Normalize to 0-1
        
        # 2. Average lifespan
        lifespans = [l.lifespan_days for l in lineages if l.lifespan_days > 0]
        avg_lifespan = np.mean(lifespans) if lifespans else 0
        lifespan_score = min(avg_lifespan / 30, 1.0)  # 30 days = max score
        
        # 3. Reinfection rate (bugs that came back after fix)
        reinfections = sum(1 for l in lineages if len(l.reintroduction_commits) > 0)
        reinfection_rate = reinfections / len(lineages) if lineages else 0
        
        # 4. Contributor spread (how many contributors touched buggy code)
        bug_contributors: Set[str] = set()
        for commit in commits:
            if commit.is_bug_fix or commit.is_bug_introducing:
                bug_contributors.add(commit.author_name)
        
        total_contributors = len(contributors) if contributors else 1
        contributor_spread = len(bug_contributors) / total_contributors
        
        # Weighted contagion score
        contagion_score = (
            propagation_depth_score * 0.25 +
            lifespan_score * 0.30 +
            reinfection_rate * 0.25 +
            contributor_spread * 0.20
        )
        
        # Interpretation
        if contagion_score < 0.3:
            interpretation = "Contained"
        elif contagion_score < 0.6:
            interpretation = "Moderate"
        else:
            interpretation = "Highly Infectious"
        
        return BugContagionScore(
            score=round(contagion_score, 3),
            interpretation=interpretation,
            propagation_depth_avg=round(avg_files_affected, 2),
            avg_lifespan_days=round(avg_lifespan, 2),
            reinfection_rate=round(reinfection_rate, 3),
            contributor_spread_factor=round(contributor_spread, 3)
        )
    
    def _identify_hotspots(self, commits: List[CommitData]) -> List[FileHotspot]:
        """Identify bug-prone files/modules"""
        file_stats: Dict[str, Dict] = defaultdict(lambda: {
            'bug_count': 0,
            'bug_fix_count': 0,
            'total_changes': 0,
            'additions': 0,
            'deletions': 0,
            'contributors': set(),
            'last_bug_date': ''
        })
        
        for commit in commits:
            for file in commit.files:
                stats = file_stats[file.filename]
                stats['total_changes'] += 1
                stats['additions'] += file.additions
                stats['deletions'] += file.deletions
                stats['contributors'].add(commit.author_name)
                
                if commit.is_bug_fix:
                    stats['bug_fix_count'] += 1
                    stats['last_bug_date'] = commit.author_date
                
                if commit.is_bug_introducing:
                    stats['bug_count'] += 1
        
        hotspots = []
        for filename, stats in file_stats.items():
            if stats['bug_fix_count'] > 0 or stats['bug_count'] > 0:
                # Calculate churn score
                total_lines = stats['additions'] + stats['deletions']
                churn_score = stats['deletions'] / stats['additions'] if stats['additions'] > 0 else 0
                
                # Calculate hotspot intensity
                bug_density = (stats['bug_count'] + stats['bug_fix_count']) / max(stats['total_changes'], 1)
                hotspot_intensity = min(bug_density * 2, 1.0)  # Scale to 0-1
                
                hotspots.append(FileHotspot(
                    filename=filename,
                    bug_count=stats['bug_count'],
                    bug_fix_count=stats['bug_fix_count'],
                    churn_score=round(churn_score, 3),
                    hotspot_intensity=round(hotspot_intensity, 3),
                    last_bug_date=stats['last_bug_date'],
                    contributors_involved=list(stats['contributors'])
                ))
        
        # Sort by hotspot intensity
        hotspots.sort(key=lambda h: h.hotspot_intensity, reverse=True)
        return hotspots[:20]  # Top 20 hotspots
    
    def _identify_super_spreaders(self, commits: List[CommitData]) -> List[SuperSpreaderCommit]:
        """Identify commits that introduced bugs affecting many files"""
        super_spreaders = []
        
        for commit in commits:
            if commit.is_bug_introducing or (commit.files_changed > 10 and commit.commit_type == 'feature'):
                # Check if this commit's files had subsequent bug fixes
                files_with_bugs = len([f for f in commit.files if f.status != 'removed'])
                
                if files_with_bugs > 3:
                    amplification = files_with_bugs / 10  # Normalize
                    super_spreaders.append(SuperSpreaderCommit(
                        sha=commit.sha,
                        message=commit.message[:100],
                        author=commit.author_name,
                        files_affected=files_with_bugs,
                        bugs_introduced=1 if commit.is_bug_introducing else 0,
                        amplification_score=round(min(amplification, 1.0), 3)
                    ))
        
        # Sort by amplification score
        super_spreaders.sort(key=lambda s: s.amplification_score, reverse=True)
        return super_spreaders[:10]
    
    def _calculate_recovery_metrics(
        self,
        commits: List[CommitData],
        lineages: List[BugLineage]
    ) -> RecoveryMetrics:
        """Calculate recovery and immune response metrics"""
        
        # Time to fix calculation
        fix_times = []
        for lineage in lineages:
            if lineage.is_resolved and lineage.lifespan_days > 0:
                fix_times.append(lineage.lifespan_days * 24)  # Convert to hours
        
        avg_time_to_fix = np.mean(fix_times) if fix_times else 0
        
        # Regression probability
        regressions = sum(1 for l in lineages if len(l.reintroduction_commits) > 0)
        regression_prob = regressions / len(lineages) if lineages else 0
        
        # Clean commit streak (consecutive non-bug commits)
        max_streak = 0
        current_streak = 0
        for commit in reversed(commits):  # Start from most recent
            if not commit.is_bug_fix and not commit.is_bug_introducing:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        # Immunity score (0-100)
        # Higher = better recovery, lower regression, longer clean streaks
        immunity_score = (
            (1 - min(avg_time_to_fix / 168, 1)) * 30 +  # 168 hours = 1 week
            (1 - regression_prob) * 40 +
            min(max_streak / 20, 1) * 30
        )
        
        # Resilience classification
        if immunity_score >= 75:
            resilience = "Antifragile"
        elif immunity_score >= 50:
            resilience = "Resilient"
        elif immunity_score >= 25:
            resilience = "Moderate"
        else:
            resilience = "Fragile"
        
        return RecoveryMetrics(
            avg_time_to_fix_hours=round(avg_time_to_fix, 2),
            regression_probability=round(regression_prob, 3),
            clean_commit_streak=max_streak,
            immunity_score=round(immunity_score, 2),
            resilience_class=resilience
        )
    
    def _analyze_developer_influence(self, commits: List[CommitData]) -> List[DeveloperInfluence]:
        """Analyze developer patterns (systemic, not personal blame)"""
        dev_stats: Dict[str, Dict] = defaultdict(lambda: {
            'commits_total': 0,
            'bug_introductions': 0,
            'bug_fixes': 0,
            'regression_associations': 0
        })
        
        for commit in commits:
            author = commit.author_name
            dev_stats[author]['commits_total'] += 1
            
            if commit.is_bug_fix:
                dev_stats[author]['bug_fixes'] += 1
            
            if commit.is_bug_introducing:
                dev_stats[author]['bug_introductions'] += 1
        
        influences = []
        for contributor, stats in dev_stats.items():
            # Net impact: positive = more fixes than introductions
            net_impact = (stats['bug_fixes'] - stats['bug_introductions']) / max(stats['commits_total'], 1)
            
            influences.append(DeveloperInfluence(
                contributor=contributor,
                commits_total=stats['commits_total'],
                bug_introductions=stats['bug_introductions'],
                bug_fixes=stats['bug_fixes'],
                regression_associations=stats['regression_associations'],
                net_impact_score=round(net_impact, 3)
            ))
        
        # Sort by net impact
        influences.sort(key=lambda d: d.net_impact_score, reverse=True)
        return influences
    
    def _detect_temporal_waves(self, commits: List[CommitData]) -> List[TemporalBugWave]:
        """Detect bug surges after major events"""
        waves = []
        
        # Look for large commits (potential refactors/releases)
        for i, commit in enumerate(commits):
            is_major_event = (
                commit.files_changed > 15 or
                'release' in commit.message.lower() or
                'refactor' in commit.message.lower() or
                'merge' in commit.message.lower()
            )
            
            if is_major_event:
                # Count bugs in subsequent commits
                subsequent = commits[i+1:i+15]
                bug_count = sum(1 for c in subsequent if c.is_bug_fix)
                
                if bug_count >= 3:  # Significant wave
                    # Determine trigger type
                    if 'release' in commit.message.lower():
                        trigger = 'release'
                    elif 'refactor' in commit.message.lower():
                        trigger = 'refactor'
                    else:
                        trigger = 'major_change'
                    
                    # Find wave end (when bugs stop)
                    wave_end_idx = i + 1
                    for j, c in enumerate(subsequent):
                        if c.is_bug_fix:
                            wave_end_idx = i + 1 + j
                    
                    # Calculate recovery duration
                    try:
                        start_date = datetime.fromisoformat(commit.author_date.replace('Z', '+00:00'))
                        end_date = datetime.fromisoformat(commits[wave_end_idx].author_date.replace('Z', '+00:00'))
                        recovery_days = (end_date - start_date).total_seconds() / 86400
                    except:
                        recovery_days = 0
                    
                    waves.append(TemporalBugWave(
                        trigger_event=trigger,
                        trigger_commit=commit.sha,
                        wave_start=commit.author_date,
                        wave_end=commits[wave_end_idx].author_date if wave_end_idx < len(commits) else commit.author_date,
                        bug_count=bug_count,
                        recovery_duration_days=round(recovery_days, 2)
                    ))
        
        return waves[:5]  # Top 5 waves


bug_evolution_engine = BugEvolutionEngine()


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
        forks: List[ForkData],
        bug_evolution: Optional[BugEvolutionAnalysis] = None
    ) -> HealthScore:
        """Calculate comprehensive health score (0-100) with bug evolution factors"""
        
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
            if n > 0 and total_contributions > 0:
                gini = (2 * sum((i + 1) * x for i, x in enumerate(sorted_contrib))) / (n * total_contributions) - (n + 1) / n
                contributor_diversity_score = 25 * (1 - abs(gini))  # Lower Gini = better diversity
            else:
                contributor_diversity_score = 12.5
        else:
            contributor_diversity_score = 5.0
        
        # 4. Change Volatility Score (25 points)
        if total_commits > 5:
            recent_commits = commits[:10]  # Last 10 commits
            recent_bug_ratio = sum(1 for c in recent_commits if c.is_bug_fix) / len(recent_commits)
            change_volatility_score = max(0, 25 - (recent_bug_ratio * 50))
        else:
            change_volatility_score = 15.0
        
        # 5. Bug Evolution Adjustments
        contagion_penalty = 0.0
        recovery_bonus = 0.0
        
        if bug_evolution:
            # Contagion penalty (up to -15 points)
            contagion_penalty = bug_evolution.contagion_score.score * 15
            
            # Recovery bonus (up to +10 points)
            recovery_bonus = (bug_evolution.recovery_metrics.immunity_score / 100) * 10
        
        # Overall score
        overall_score = (
            bug_frequency_score +
            commit_stability_score +
            contributor_diversity_score +
            change_volatility_score -
            contagion_penalty +
            recovery_bonus
        )
        
        # Clamp to 0-100
        overall_score = max(0, min(100, overall_score))
        
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
            contagion_penalty=round(contagion_penalty, 2),
            recovery_bonus=round(recovery_bonus, 2),
            risk_level=risk_level
        )
    
    def compare_projects(
        self,
        repo_a: RepositoryData,
        health_a: HealthScore,
        commits_a: List[CommitData],
        bug_evolution_a: Optional[BugEvolutionAnalysis],
        repo_b: RepositoryData,
        health_b: HealthScore,
        commits_b: List[CommitData],
        bug_evolution_b: Optional[BugEvolutionAnalysis]
    ) -> ComparisonResult:
        """Compare two projects and determine the healthier one"""
        
        metrics = {
            "repo_a": {
                "health_score": health_a.overall_score,
                "bug_ratio": sum(1 for c in commits_a if c.is_bug_fix) / len(commits_a) if commits_a else 0,
                "total_commits": len(commits_a),
                "risk_level": health_a.risk_level,
                "contagion_score": bug_evolution_a.contagion_score.score if bug_evolution_a else 0,
                "immunity_score": bug_evolution_a.recovery_metrics.immunity_score if bug_evolution_a else 0
            },
            "repo_b": {
                "health_score": health_b.overall_score,
                "bug_ratio": sum(1 for c in commits_b if c.is_bug_fix) / len(commits_b) if commits_b else 0,
                "total_commits": len(commits_b),
                "risk_level": health_b.risk_level,
                "contagion_score": bug_evolution_b.contagion_score.score if bug_evolution_b else 0,
                "immunity_score": bug_evolution_b.recovery_metrics.immunity_score if bug_evolution_b else 0
            }
        }
        
        # Bug evolution comparison
        bug_evolution_comparison = {}
        if bug_evolution_a and bug_evolution_b:
            bug_evolution_comparison = {
                "contagion_winner": repo_a.full_name if bug_evolution_a.contagion_score.score < bug_evolution_b.contagion_score.score else repo_b.full_name,
                "recovery_winner": repo_a.full_name if bug_evolution_a.recovery_metrics.immunity_score > bug_evolution_b.recovery_metrics.immunity_score else repo_b.full_name,
                "hotspot_count_a": len(bug_evolution_a.file_hotspots),
                "hotspot_count_b": len(bug_evolution_b.file_hotspots),
                "active_bugs_a": bug_evolution_a.active_bugs,
                "active_bugs_b": bug_evolution_b.active_bugs
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
        
        contagion_note = ""
        if bug_evolution_a and bug_evolution_b:
            winner_contagion = winner_metrics["contagion_score"]
            if winner_contagion < 0.3:
                contagion_note = "Bug spread is well contained. "
            elif winner_contagion < 0.6:
                contagion_note = "Bug spread is moderate but manageable. "
            else:
                contagion_note = "Warning: High bug contagion detected. "
        
        explanation = (
            f"{winner} is healthier with a score of {winner_health.overall_score:.1f}/100 "
            f"compared to {loser_health.overall_score:.1f}/100. "
            f"It has {bug_comparison}, "
            f"{'stable' if winner_health.commit_stability_score > 18 else 'moderate'} update patterns, "
            f"and {winner_health.risk_level.lower()} risk level. "
            f"{contagion_note}"
            f"The {score_diff:.1f} point advantage suggests "
            f"{'significantly' if score_diff > 20 else 'moderately'} better long-term maintainability."
        )
        
        return ComparisonResult(
            repo_a=repo_a.full_name,
            repo_b=repo_b.full_name,
            winner=winner,
            explanation=explanation,
            metrics=metrics,
            bug_evolution_comparison=bug_evolution_comparison
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
    """Analyze a GitHub repository with full bug evolution analysis"""
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
                    try:
                        last_analyzed = datetime.fromisoformat(cached['last_analyzed'].replace('Z', '+00:00'))
                        if datetime.now(timezone.utc) - last_analyzed < timedelta(hours=1):
                            return cached
                    except:
                        pass
        
        # Fetch data
        repo_data = await github_service.fetch_repository_info(owner, repo)
        commits, contributors, forks = await asyncio.gather(
            github_service.fetch_commits(owner, repo),
            github_service.fetch_contributors(owner, repo),
            github_service.fetch_forks(owner, repo)
        )
        
        # Calculate DNA features
        dna_features = analysis_engine.calculate_dna_features(commits)
        
        # Perform bug evolution analysis
        bug_evolution = bug_evolution_engine.analyze_bug_evolution(
            commits, contributors, forks, repo_data.id
        )
        
        # Calculate health score with bug evolution factors
        health_score = analysis_engine.calculate_health_score(
            repo_data, commits, contributors, forks, bug_evolution
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
            "bug_evolution": bug_evolution.model_dump(),
            "analysis_summary": {
                "total_commits": len(commits),
                "bug_fixes": sum(1 for c in commits if c.is_bug_fix),
                "bug_introducing": sum(1 for c in commits if c.is_bug_introducing),
                "total_contributors": len(contributors),
                "total_forks": len(forks),
                "active_bugs": bug_evolution.active_bugs,
                "resolved_bugs": bug_evolution.resolved_bugs,
                "contagion_level": bug_evolution.contagion_score.interpretation,
                "resilience_class": bug_evolution.recovery_metrics.resilience_class
            },
            "last_analyzed": datetime.now(timezone.utc).isoformat()
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
    """Compare two GitHub repositories with bug evolution analysis"""
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
        
        # Parse bug evolution data
        bug_evolution_a = BugEvolutionAnalysis(**analysis_a['bug_evolution']) if 'bug_evolution' in analysis_a else None
        bug_evolution_b = BugEvolutionAnalysis(**analysis_b['bug_evolution']) if 'bug_evolution' in analysis_b else None
        
        # Compare
        comparison = analysis_engine.compare_projects(
            repo_a, health_a, commits_a, bug_evolution_a,
            repo_b, health_b, commits_b, bug_evolution_b
        )
        
        result = {
            "comparison": comparison.model_dump(),
            "repo_a_analysis": analysis_a,
            "repo_b_analysis": analysis_b
        }
        
        # Store comparison
        await db.comparisons.insert_one({
            **result,
            "compared_at": datetime.now(timezone.utc).isoformat()
        })
        
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


@api_router.get("/bug-evolution/{owner}/{repo}")
async def get_bug_evolution(owner: str, repo: str):
    """Get bug evolution analysis for a repository"""
    cached = await db.repositories.find_one(
        {"owner": owner, "repo": repo},
        {"_id": 0, "bug_evolution": 1}
    )
    
    if not cached or 'bug_evolution' not in cached:
        raise HTTPException(status_code=404, detail="Repository not analyzed yet or no bug evolution data")
    
    return cached.get('bug_evolution', {})


@api_router.get("/hotspots/{owner}/{repo}")
async def get_hotspots(owner: str, repo: str):
    """Get file hotspots for a repository"""
    cached = await db.repositories.find_one(
        {"owner": owner, "repo": repo},
        {"_id": 0, "bug_evolution.file_hotspots": 1}
    )
    
    if not cached or 'bug_evolution' not in cached:
        raise HTTPException(status_code=404, detail="Repository not analyzed yet")
    
    return cached.get('bug_evolution', {}).get('file_hotspots', [])


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