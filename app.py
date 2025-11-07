from flask import Flask, render_template, request, redirect, url_for, session, flash, g, make_response, jsonify, send_from_directory, Response
import sqlite3
import os
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from user_agents import parse
import pytz
import uuid
from functools import wraps
from flask_caching import Cache
from flask_compress import Compress
import time
import random
import math
import requests
import shutil


class RemoteUploadError(Exception):
    """Raised when uploading a file to remote storage fails."""

    def __init__(self, message='Upload failed.', status_code=None):
        super().__init__(message)
        self.status_code = status_code


class RemoteUploadTooLargeError(RemoteUploadError):
    """Raised when the remote server rejects the upload due to size (HTTP 413)."""

import requests

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))


# Serve favicon at /favicon.ico to satisfy browser requests
@app.route('/favicon.ico')
def favicon():
    # Return the favicon file from the static folder. Using jpeg is fine — browsers accept it.
    return send_from_directory(app.static_folder, 'favicon.jpg', mimetype='image/jpeg')

# Enable response compression (brotli preferred, falls back when not available)
app.config.setdefault('COMPRESS_ALGORITHM', 'br')  # use 'br' (brotli) when available; falls back based on client/server support
app.config.setdefault('COMPRESS_LEVEL', 6)
app.config.setdefault('COMPRESS_MIN_SIZE', 500)
Compress(app)

# Cache configuration
cache_config = {'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300}
redis_url = os.environ.get('REDIS_URL') or os.environ.get('REDIS_URLS')
if redis_url:
    # Use Redis if available (set REDIS_URL in environment)
    cache_config = {
        'CACHE_TYPE': 'RedisCache',
        'CACHE_REDIS_URL': redis_url,
        'CACHE_DEFAULT_TIMEOUT': 300
    }
cache = Cache(config=cache_config)
cache.init_app(app)

# Rate limiting configuration
RATE_LIMIT = 100  # requests
RATE_LIMIT_WINDOW = 60  # seconds
ip_request_times = {}

# Admin credentials from environment
ADMIN_USER = os.environ.get('ADMIN_USER', 'admin09')
ADMIN_PASS = os.environ.get('ADMIN_PASS', '#admin1234')

# Rate limiting decorator
def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        current_time = time.time()
        
        if ip not in ip_request_times:
            ip_request_times[ip] = []
        
        ip_request_times[ip] = [t for t in ip_request_times[ip] if current_time - t < RATE_LIMIT_WINDOW]
        
        if len(ip_request_times[ip]) >= RATE_LIMIT:
            response = make_response("Too many requests. Please try again later.", 429)
            response.headers['Retry-After'] = RATE_LIMIT_WINDOW
            return response
        
        ip_request_times[ip].append(current_time)
        return f(*args, **kwargs)
    return decorated_function

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'images')
app.config['TRAILER_FOLDER'] = os.path.join('static', 'trailers')
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'webm'}
app.config['DATABASE'] = 'movies.db'
app.config['MAX_FEATURED'] = 5
app.config['ANALYTICS_DB'] = 'analytics.db'
app.config['ALL_GENRES'] = [
   'Amateur', 'Anal', 'BDSM', 'Blowjob', 'Creampie', 
    'Ebony', 'Lesbian', 'Gay', 'MILF', 'Teen', 
    'Group', 'Compilation', 'Cosplay', 'Threesome', 'Asian', 
    'POV', 'Hardcore', 'Softcore', 'Vintage', 'Web Series', "AsianM",
    'Desi',"BRZ", "ANIME" 
]
app.config['REMOTE_UPLOAD_URL'] = os.environ.get('REMOTE_UPLOAD_URL', 'http://217.217.249.3:8000/upload')

# Fixed API token for protecting /api routes (change this to a strong token)
app.config['API_TOKEN'] = os.environ.get('API_TOKEN', "my_secret_token_12345")

# Create upload directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAILER_FOLDER'], exist_ok=True)

def migrate_database(conn):
    """Handle all database migrations automatically"""
    try:
        # Check if we need to migrate from old schema to new schema
        cursor = conn.execute("PRAGMA table_info(movies)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Migration 1: Add links column if it doesn't exist
        if 'links' not in columns:
            conn.execute('ALTER TABLE movies ADD COLUMN links TEXT')
            # Migrate data from link to links if link column exists
            if 'link' in columns:
                movies = conn.execute('SELECT id, link FROM movies WHERE link IS NOT NULL').fetchall()
                for movie in movies:
                    conn.execute('UPDATE movies SET links = ? WHERE id = ?', (movie['link'], movie['id']))
            conn.commit()
        
        # Migration 2: Remove the old link column if it exists (SQLite doesn't support DROP COLUMN directly)
        if 'link' in columns:
            # Create a new table without the link column
            conn.execute('''
                CREATE TABLE IF NOT EXISTS movies_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    image TEXT NOT NULL,
                    links TEXT NOT NULL,
                    genres TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    trailer TEXT DEFAULT NULL,
                    is_featured INTEGER DEFAULT 0,
                    year INTEGER DEFAULT 2023,
                    language TEXT DEFAULT 'Hindi',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Copy data to new table
            conn.execute('''
                INSERT INTO movies_new (id, title, image, links, genres, description, 
                                      trailer, is_featured, year, language, created_at)
                SELECT id, title, image, COALESCE(links, link), genres, description, 
                       trailer, is_featured, year, language, created_at
                FROM movies
            ''')
            
            # Drop old table and rename new one
            conn.execute('DROP TABLE movies')
            conn.execute('ALTER TABLE movies_new RENAME TO movies')
            
            # Recreate indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_created ON movies (created_at DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_title ON movies (title)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_genres ON movies (genres)')
            
            conn.commit()
            
    except Exception as e:
        print(f"Database migration error: {str(e)}")
        conn.rollback()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
        # Run migrations
        migrate_database(g.db)
    return g.db

def get_analytics_db():
    if 'analytics_db' not in g:
        g.analytics_db = sqlite3.connect(app.config['ANALYTICS_DB'])
        g.analytics_db.row_factory = sqlite3.Row
    return g.analytics_db

def init_db():
    with app.app_context():
        conn = sqlite3.connect(app.config['DATABASE'])
        try:
            # Create the table with the new schema
            conn.execute('''
                CREATE TABLE IF NOT EXISTS movies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    image TEXT NOT NULL,
                    links TEXT NOT NULL,
                    genres TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    trailer TEXT DEFAULT NULL,
                    is_featured INTEGER DEFAULT 0,
                    year INTEGER DEFAULT 2023,
                    language TEXT DEFAULT 'Hindi',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_created ON movies (created_at DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_title ON movies (title)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_movies_genres ON movies (genres)')

            # Ensure a simple page_views table exists in the main movies.db as a fallback
            conn.execute('''
                CREATE TABLE IF NOT EXISTS page_views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT,
                    ip TEXT,
                    user_agent TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
        finally:
            conn.close()

def init_analytics_db():
    with app.app_context():
        conn = sqlite3.connect(app.config['ANALYTICS_DB'])
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS page_views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_url TEXT NOT NULL,
                    user_agent TEXT,
                    device_type TEXT,
                    device_model TEXT,
                    browser TEXT,
                    os TEXT,
                    os_version TEXT,
                    referrer TEXT,
                    button_clicked INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    visitor_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_traffic (
                    date DATE PRIMARY KEY,
                    visits INTEGER DEFAULT 0,
                    unique_visitors INTEGER DEFAULT 0,
                    returning_visitors INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS button_clicks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_url TEXT NOT NULL,
                    button_type TEXT NOT NULL,
                    movie_id INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS unique_visitors (
                    visitor_id TEXT PRIMARY KEY,
                    first_visit DATETIME,
                    last_visit DATETIME,
                    visit_count INTEGER DEFAULT 1,
                    device_type TEXT,
                    device_model TEXT,
                    browser TEXT,
                    os TEXT
                )
            ''')
            
            # New table: stores short-lived active sessions for "active users" count
            conn.execute('''
                CREATE TABLE IF NOT EXISTS active_sessions (
                    visitor_id TEXT PRIMARY KEY,
                    ip TEXT,
                    user_agent TEXT,
                    last_seen DATETIME
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_visitor_id ON page_views (visitor_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_page_views_date ON page_views (date(timestamp))')
            
            conn.commit()
        finally:
            conn.close()

def allowed_file(filename, file_type):
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        return ext in app.config['ALLOWED_VIDEO_EXTENSIONS']
    return False

# small helper specifically for image files
def allowed_image_file(filename):
    return allowed_file(filename, 'image')


def guess_video_mime(url):
    """Best-effort guess of the video MIME type based on file extension."""
    if not url:
        return 'video/mp4'
    lowered = url.lower()
    if lowered.endswith('.mkv'):
        return 'video/x-matroska'
    if lowered.endswith('.webm'):
        return 'video/webm'
    if lowered.endswith('.mov'):
        return 'video/quicktime'
    if lowered.endswith('.m4v'):
        return 'video/x-m4v'
    return 'video/mp4'


def upload_to_remote_storage(file_storage):
    """Send file to remote FastAPI storage and return the public URL."""
    if not file_storage or not file_storage.filename:
        return None

    upload_endpoint = app.config.get('REMOTE_UPLOAD_URL')
    if not upload_endpoint:
        app.logger.error('REMOTE_UPLOAD_URL is not configured.')
        return None

    filename = secure_filename(file_storage.filename)
    stream = getattr(file_storage, 'stream', file_storage)
    if hasattr(stream, 'seek'):
        try:
            stream.seek(0)
        except Exception:
            pass
    try:
        response = requests.post(
            upload_endpoint,
            files={'file': (filename, stream, file_storage.mimetype or 'application/octet-stream')},
            timeout=120
        )
        response.raise_for_status()
        payload = response.json()
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        app.logger.error('Remote upload failed for %s with status %s: %s', filename, status_code, exc)
        if status_code == 413:
            raise RemoteUploadTooLargeError('Remote storage rejected the upload because the file is too large.', status_code)
        raise RemoteUploadError('Remote storage returned an error.', status_code) from exc
    except (requests.RequestException, ValueError) as exc:
        app.logger.error('Remote upload failed for %s: %s', filename, exc)
        raise RemoteUploadError('Unable to reach remote storage.') from exc

    url = payload.get('url')
    if not url:
        app.logger.error('Remote upload response missing url for %s: %s', filename, payload)
        raise RemoteUploadError('Remote storage did not return a file URL.')

    return url

def get_kolkata_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def parse_user_agent(user_agent_string):
    """Parse user agent string and return device information"""
    ua = parse(user_agent_string)
    
    device_type = 'Desktop'
    if ua.is_mobile:
        device_type = 'Mobile'
    elif ua.is_tablet:
        device_type = 'Tablet'
    
    device_model = 'Unknown'
    if ua.device.family != 'Other':
        device_model = ua.device.family
    
    browser = ua.browser.family if ua.browser.family != 'Other' else 'Unknown'
    os = ua.os.family if ua.os.family != 'Other' else 'Unknown'
    os_version = ua.os.version_string if ua.os.version_string else 'Unknown'
    
    return {
        'device_type': device_type,
        'device_model': device_model,
        'browser': browser,
        'os': os,
        'os_version': os_version
    }

@app.before_request
def before_request():
    # Skip analytics logging for static assets like robots.txt to avoid noise
    if request.path == '/robots.txt':
        return
    log_page_view()

def log_page_view():
    if request.path.startswith('/static/') or request.path.startswith('/admin'):
        return
        
    with get_analytics_db() as conn:
        user_agent = request.user_agent.string
        referrer = request.referrer or ''
        ua_info = parse_user_agent(user_agent)
        kolkata_time = get_kolkata_time()
        today = kolkata_time.strftime('%Y-%m-%d')
        
        visitor_id = request.cookies.get('visitor_id')
        is_new_visitor = False
        
        if not visitor_id:
            visitor_id = str(uuid.uuid4())
            is_new_visitor = True
        
        button_clicked = 0
        if request.args.get('action') == 'watch':
            button_clicked = 1
            movie_id = request.args.get('movie_id')
            conn.execute('''
                INSERT INTO button_clicks (page_url, button_type, movie_id)
                VALUES (?, ?, ?)
            ''', (request.path, 'watch', movie_id))
        
        conn.execute('''
            INSERT INTO page_views 
            (page_url, user_agent, device_type, device_model, 
             browser, os, os_version, referrer, button_clicked, timestamp, 
             visitor_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.path, user_agent,
            ua_info['device_type'], ua_info['device_model'],
            ua_info['browser'], ua_info['os'], ua_info['os_version'],
            referrer, button_clicked, kolkata_time.strftime('%Y-%m-%d %H:%M:%S'),
            visitor_id
        ))
        
        if is_new_visitor:
            conn.execute('''
                INSERT INTO unique_visitors 
                (visitor_id, first_visit, last_visit, visit_count,
                 device_type, device_model, browser, os)
                VALUES (?, ?, ?, 1, ?, ?, ?, ?)
            ''', (
                visitor_id, 
                kolkata_time.strftime('%Y-%m-%d %H:%M:%S'),
                kolkata_time.strftime('%Y-%m-%d %H:%M:%S'),
                ua_info['device_type'], ua_info['device_model'],
                ua_info['browser'], ua_info['os']
            ))
            # Store visitor ID to set cookie in after_request
            g._new_visitor_id = visitor_id
        else:
            conn.execute('''
                UPDATE unique_visitors 
                SET last_visit = ?, visit_count = visit_count + 1
                WHERE visitor_id = ?
            ''', (kolkata_time.strftime('%Y-%m-%d %H:%M:%S'), visitor_id))
        
        conn.execute('''
            INSERT OR IGNORE INTO daily_traffic (date, visits, unique_visitors, returning_visitors)
            VALUES (?, 0, 0, 0)
        ''', (today,))
        
        conn.execute('''
            UPDATE daily_traffic SET visits = visits + 1 WHERE date = ?
        ''', (today,))
        
        if is_new_visitor:
            conn.execute('''
                UPDATE daily_traffic SET unique_visitors = unique_visitors + 1 WHERE date = ?
            ''', (today,))
        else:
            conn.execute('''
                UPDATE daily_traffic SET returning_visitors = returning_visitors + 1 WHERE date = ?
            ''', (today,))
        
        conn.commit()

        # Maintain short-lived active_sessions table (use UTC timestamps here)
        try:
            last_seen_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            conn.execute('''
                INSERT OR REPLACE INTO active_sessions (visitor_id, ip, user_agent, last_seen)
                VALUES (?, ?, ?, ?)
            ''', (visitor_id, request.remote_addr, user_agent, last_seen_utc))
            conn.commit()
        except Exception as e:
            print(f"Failed to update active_sessions: {e}")

@app.after_request
def set_visitor_cookie(response):
    vid = getattr(g, '_new_visitor_id', None)
    if vid:
        response.set_cookie(
            'visitor_id',
            value=vid,
            max_age=31536000,
            httponly=True,
            secure=True if request.is_secure else False,
            samesite="Lax"
        )
    # Add long-lived cache headers for static assets served by Flask.
    # In production, serve static assets with a real web server (nginx/Caddy) and keep these headers there.
    if request.path.startswith('/static/') and response.status_code == 200:
        # Only set if not already set by some middleware
        response.headers.setdefault('Cache-Control', 'public, max-age=86400, immutable')
        try:
            expires = (datetime.utcnow() + timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
            response.headers.setdefault('Expires', expires)
        except Exception:
            pass
    return response

@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'db'):
        g.db.close()
    if hasattr(g, 'analytics_db'):
        g.analytics_db.close()

@app.route('/')
@rate_limit
@cache.cached(timeout=600)
def home():
    with get_db() as conn:
        featured_movies = conn.execute('''
            SELECT * FROM movies WHERE is_featured = 1 ORDER BY created_at DESC LIMIT ?
        ''', (app.config['MAX_FEATURED'],)).fetchall()
        
        if not featured_movies:
            featured_movies = conn.execute('''
                SELECT * FROM movies ORDER BY created_at DESC LIMIT ?
            ''', (app.config['MAX_FEATURED'],)).fetchall()
        
        featured_movie = featured_movies[0] if featured_movies else None
        movies = conn.execute('SELECT * FROM movies ORDER BY created_at DESC').fetchall()
    
    genre_dict = {}
    for movie in movies:
        genres = movie['genres'].split(',')
        primary_genre = genres[0].strip() if genres else 'Other'
        
        if primary_genre not in genre_dict:
            genre_dict[primary_genre] = []
        movie_dict = dict(movie)
        movie_dict['genres_list'] = [genre.strip() for genre in genres]
        genre_dict[primary_genre].append(movie_dict)
    
    return_value = render_template(
        'home.html',
        genre_dict=genre_dict,
        featured_movie=dict(featured_movie) if featured_movie else None,
        featured_movies=[dict(m) for m in featured_movies]
    )
    resp = make_response(return_value)
    resp.headers.setdefault('Cache-Control', 'public, max-age=600, stale-while-revalidate=60')
    return resp

@app.route('/movie/<int:movie_id>')
@rate_limit
@cache.cached(timeout=300, query_string=True)
def movie_detail(movie_id):
    with get_db() as conn:
        movie = conn.execute('SELECT * FROM movies WHERE id = ?', (movie_id,)).fetchone()
        
        if movie:
            primary_genre = movie['genres'].split(',')[0].strip()
            related_movies = conn.execute('''
                SELECT id, title, image FROM movies 
                WHERE genres LIKE ? AND id != ?
                ORDER BY RANDOM() 
                LIMIT 50
            ''', (f'%{primary_genre}%', movie_id)).fetchall()
            
            movie_dict = dict(movie)
            movie_dict['genres_list'] = [genre.strip() for genre in movie['genres'].split(',')]
            movie_dict['links_list'] = [link.strip() for link in movie['links'].split(',') if link.strip()] if movie['links'] else []
            
            # Get the first video URL for the player
            first_video_url = movie_dict['links_list'][0] if movie_dict['links_list'] else None
            
            movie_dict['episode_sources'] = [
                {
                    'number': idx + 1,
                    'url': link,
                    'mime': guess_video_mime(link)
                }
                for idx, link in enumerate(movie_dict['links_list']) if link
            ]
            
            # Add genre previews for all genres
            genre_counts = {}
            genre_previews = {}
            for genre in app.config['ALL_GENRES']:
                count = conn.execute('SELECT COUNT(*) FROM movies WHERE genres LIKE ?', (f'%{genre}%',)).fetchone()[0]
                if count > 0:
                    genre_counts[genre] = count
                    previews = conn.execute('SELECT id, title, image FROM movies WHERE genres LIKE ? ORDER BY RANDOM() LIMIT 3', (f'%{genre}%',)).fetchall()
                    genre_previews[genre] = [dict(m) for m in previews]
            
            rendered = render_template('movie_detail.html', 
                                movie=movie_dict,
                                video_url=first_video_url,
                                related_movies=[dict(m) for m in related_movies],
                                genre_previews=genre_previews,
                                genre_counts=genre_counts)
            resp = make_response(rendered)
            resp.headers.setdefault('Cache-Control', 'public, max-age=300, stale-while-revalidate=60')
            return resp
    
    flash('Movie not found', 'error')
    return redirect(url_for('home'))

@app.route('/admin', methods=['GET', 'POST'])
@rate_limit
def admin():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    page = request.args.get('page', 1, type=int)
    per_page = 10
    search_query = request.args.get('search', '').strip()

    if request.method == 'POST':
        try:
            title = request.form.get('title', '').strip()
            links = [link.strip() for link in request.form.getlist('links[]') if link.strip()]
            genres = request.form.getlist('genres')
            new_genre = request.form.get('new_genre', '').strip()
            description = request.form.get('description', '').strip()
            year = int(request.form.get('year', 2023))
            language = request.form.get('language', 'Hindi').strip()
            is_featured = 1 if request.form.get('is_featured') else 0
            image = request.files.get('image')
            trailer = request.files.get('trailer')
            uploaded_movie = request.files.get('file')

            uploaded_links = []
            if uploaded_movie and uploaded_movie.filename:
                if not allowed_file(uploaded_movie.filename, 'video'):
                    flash('Invalid video file type', 'error')
                    return redirect(url_for('admin'))

                try:
                    remote_video_url = upload_to_remote_storage(uploaded_movie)
                    uploaded_links.append(remote_video_url)
                except RemoteUploadTooLargeError:
                    flash('Upload failed: file is too large for remote storage. Please use a smaller file or contact support.', 'error')
                    return redirect(url_for('admin'))
                except RemoteUploadError as exc:
                    flash(str(exc), 'error')
                    return redirect(url_for('admin'))

            if new_genre and new_genre not in app.config['ALL_GENRES']:
                app.config['ALL_GENRES'].append(new_genre)
                genres.append(new_genre)

            if not genres and new_genre:
                genres = [new_genre]

            if not title or not (links or uploaded_links) or not genres:
                flash('Please fill all required fields', 'error')
                return redirect(url_for('admin'))

            # Validate image
            if not image or not image.filename:
                flash('Image is required', 'error')
                return redirect(url_for('admin'))

            if not allowed_file(image.filename, 'image'):
                flash('Invalid image file type', 'error')
                return redirect(url_for('admin'))

            try:
                image_url = upload_to_remote_storage(image)
            except RemoteUploadTooLargeError:
                flash('Poster upload failed: file is too large for remote storage.', 'error')
                return redirect(url_for('admin'))
            except RemoteUploadError as exc:
                flash(str(exc), 'error')
                return redirect(url_for('admin'))

            trailer_url = None
            if trailer and trailer.filename:
                if not allowed_file(trailer.filename, 'video'):
                    flash('Invalid video file type', 'error')
                    return redirect(url_for('admin'))
                try:
                    trailer_url = upload_to_remote_storage(trailer)
                except RemoteUploadTooLargeError:
                    flash('Trailer upload failed: file is too large for remote storage.', 'error')
                    return redirect(url_for('admin'))
                except RemoteUploadError as exc:
                    flash(str(exc), 'error')
                    return redirect(url_for('admin'))

            links = uploaded_links + links
            genres_str = ', '.join(genres)
            links_str = ', '.join(links)

            with get_db() as conn:
                if is_featured:
                    current_featured = conn.execute('SELECT COUNT(*) FROM movies WHERE is_featured = 1').fetchone()[0]
                    if current_featured >= app.config['MAX_FEATURED']:
                        oldest_featured = conn.execute('''
                            SELECT id FROM movies WHERE is_featured = 1 ORDER BY created_at ASC LIMIT 1
                        ''').fetchone()
                        if oldest_featured:
                            conn.execute('UPDATE movies SET is_featured = 0 WHERE id = ?', (oldest_featured['id'],))

                cursor = conn.execute('''
                    INSERT INTO movies (title, image, links, genres, description, trailer, is_featured, year, language)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (title, image_url, links_str, genres_str, description, trailer_url, is_featured, year, language))
                conn.commit()
                new_movie_id = cursor.lastrowid  # capture inserted movie id

            flash('Uploaded successfully!', 'success')
            return redirect(url_for('movie_detail', movie_id=new_movie_id))
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('admin'))

    with get_db() as conn:
        if search_query:
            movies = conn.execute('''
                SELECT * FROM movies 
                WHERE title LIKE ? OR genres LIKE ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (
                f'%{search_query}%', 
                f'%{search_query}%',
                per_page,
                (page - 1) * per_page
            )).fetchall()
            
            total_movies = conn.execute('''
                SELECT COUNT(*) FROM movies 
                WHERE title LIKE ? OR genres LIKE ?
            ''', (f'%{search_query}%', f'%{search_query}%')).fetchone()[0]
        else:
            movies = conn.execute('''
                SELECT * FROM movies 
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (per_page, (page - 1) * per_page)).fetchall()
            
            total_movies = conn.execute('SELECT COUNT(*) FROM movies').fetchone()[0]
    
    movies_with_details = []
    for movie in movies:
        movie_dict = dict(movie)
        movie_dict['genres_list'] = [genre.strip() for genre in movie['genres'].split(',')]
        movie_dict['links_list'] = [link.strip() for link in movie['links'].split(',')] if movie['links'] else []
        movies_with_details.append(movie_dict)
    
    return render_template('admin.html', 
                         movies=movies_with_details,
                         all_genres=app.config['ALL_GENRES'],
                         search_query=search_query,
                         page=page,
                         per_page=per_page,
                         total_movies=total_movies)

@app.route('/edit/<int:movie_id>', methods=['GET', 'POST'])
@rate_limit
def edit_movie(movie_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    with get_db() as conn:
        movie = conn.execute('SELECT * FROM movies WHERE id = ?', (movie_id,)).fetchone()
    
    if not movie:
        flash('Movie not found', 'error')
        return redirect(url_for('admin'))

    if request.method == 'POST':
        try:
            title = request.form.get('title', '').strip()
            links = [link.strip() for link in request.form.getlist('links[]') if link.strip()]
            genres = request.form.getlist('genres')
            new_genre = request.form.get('new_genre', '').strip()
            description = request.form.get('description', '').strip()
            year = int(request.form.get('year', 2023))
            language = request.form.get('language', 'Hindi').strip()
            is_featured = 1 if request.form.get('is_featured') else 0
            image = request.files.get('image')
            trailer = request.files.get('trailer')
            uploaded_movie = request.files.get('file')

            uploaded_links = []
            if uploaded_movie and uploaded_movie.filename:
                if not allowed_file(uploaded_movie.filename, 'video'):
                    flash('Invalid video file type', 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))

                try:
                    remote_video_url = upload_to_remote_storage(uploaded_movie)
                    uploaded_links.append(remote_video_url)
                except RemoteUploadTooLargeError:
                    flash('Upload failed: file is too large for remote storage. Please use a smaller file or contact support.', 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))
                except RemoteUploadError as exc:
                    flash(str(exc), 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))

            if new_genre and new_genre not in app.config['ALL_GENRES']:
                app.config['ALL_GENRES'].append(new_genre)
                genres.append(new_genre)

            if not genres and new_genre:
                genres = [new_genre]

            links = uploaded_links + links

            if not title or not links or not genres:
                flash('Please fill all required fields', 'error')
                return redirect(url_for('edit_movie', movie_id=movie_id))

            genres_str = ', '.join(genres)
            links_str = ', '.join(links)

            update_values = {
                'title': title,
                'links': links_str,
                'genres': genres_str,
                'description': description,
                'year': year,
                'language': language,
                'is_featured': is_featured,
                'id': movie_id
            }

            if image and image.filename:
                if not allowed_file(image.filename, 'image'):
                    flash('Invalid image file type', 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))
                try:
                    image_url = upload_to_remote_storage(image)
                except RemoteUploadTooLargeError:
                    flash('Poster upload failed: file is too large for remote storage.', 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))
                except RemoteUploadError as exc:
                    flash(str(exc), 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))
                update_values['image'] = image_url

            if trailer and trailer.filename:
                if not allowed_file(trailer.filename, 'video'):
                    flash('Invalid video file type', 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))
                try:
                    trailer_url = upload_to_remote_storage(trailer)
                except RemoteUploadTooLargeError:
                    flash('Trailer upload failed: file is too large for remote storage.', 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))
                except RemoteUploadError as exc:
                    flash(str(exc), 'error')
                    return redirect(url_for('edit_movie', movie_id=movie_id))
                update_values['trailer'] = trailer_url

            with get_db() as conn:
                if is_featured:
                    current_featured = conn.execute('SELECT COUNT(*) FROM movies WHERE is_featured = 1').fetchone()[0]
                    if current_featured >= app.config['MAX_FEATURED']:
                        oldest_featured = conn.execute('''
                            SELECT id FROM movies WHERE is_featured = 1 ORDER BY created_at ASC LIMIT 1
                        ''').fetchone()
                        if oldest_featured and oldest_featured['id'] != movie_id:
                            conn.execute('UPDATE movies SET is_featured = 0 WHERE id = ?', (oldest_featured['id'],))

                set_clause = ', '.join([f"{k} = ?" for k in update_values.keys() if k != 'id'])
                values = [v for k, v in update_values.items() if k != 'id']
                values.append(movie_id)
                
                conn.execute(f'''
                    UPDATE movies SET {set_clause} WHERE id = ?
                ''', values)
                conn.commit()

            flash('Movie updated successfully!', 'success')
            return redirect(url_for('admin'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return redirect(url_for('edit_movie', movie_id=movie_id))

    movie_dict = dict(movie)
    movie_dict['genres_list'] = [genre.strip() for genre in movie['genres'].split(',')]
    movie_dict['links_list'] = [link.strip() for link in movie['links'].split(',')] if movie['links'] else []
    return render_template('edit_movie.html', 
                         movie=movie_dict,
                         all_genres=app.config['ALL_GENRES'])

@app.route('/delete/<int:movie_id>')
@rate_limit
def delete_movie(movie_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    with get_db() as conn:
        movie = conn.execute('SELECT * FROM movies WHERE id = ?', (movie_id,)).fetchone()
        if movie:
            conn.execute('DELETE FROM movies WHERE id = ?', (movie_id,))
            conn.commit()
            flash('Movie deleted successfully', 'success')
        else:
            flash('Movie not found', 'error')
    
    return redirect(url_for('admin'))

@app.route('/tvshow')
@rate_limit
@cache.cached(timeout=300)
def tvshow():
    with get_db() as conn:
        tv_shows = conn.execute('SELECT * FROM movies WHERE genres LIKE ? ORDER BY created_at DESC', ('%TV Show%',)).fetchall()
    shows = [dict(show) for show in tv_shows]
    return render_template('tvshow.html', tv_shows=shows, movies=shows)

@app.route('/recent')
@rate_limit
@cache.cached(timeout=600)
def recent():
    with get_db() as conn:
        recent_movies = conn.execute('''
            SELECT * FROM movies 
            ORDER BY created_at DESC 
            LIMIT 20
        ''').fetchall()
    resp = make_response(render_template('recent.html', movies=[dict(movie) for movie in recent_movies]))
    resp.headers.setdefault('Cache-Control', 'public, max-age=600, stale-while-revalidate=60')
    return resp

@app.route('/search')
@rate_limit
@cache.cached(timeout=300, query_string=True)
def search():
    query = request.args.get('q', '').strip()
    
    if not query:
        return redirect(url_for('home'))

    try:
        with get_db() as conn:
            search_terms = query.split()
            where_clauses = []
            params = []
            
            for term in search_terms:
                where_clauses.append("(title LIKE ? OR description LIKE ? OR genres LIKE ?)")
                params.extend([f'%{term}%', f'%{term}%', f'%{term}%'])
            
            where_clause = " AND ".join(where_clauses)
            
            sql = f'''
                SELECT * FROM movies 
                WHERE {where_clause}
                ORDER BY created_at DESC
            '''
            
            movies = conn.execute(sql, params).fetchall()
            
            return render_template('search_results.html', 
                                movies=[dict(movie) for movie in movies],
                                query=query)
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        flash('An error occurred during search', 'error')
        return redirect(url_for('home'))

@app.route('/genre/<genre_name>')
@rate_limit
@cache.cached(timeout=120, query_string=True)
def genre(genre_name):
    # Server-side pagination
    page = request.args.get('page', 1, type=int)
    per_page = 60
    if page < 1:
        page = 1
    offset = (page - 1) * per_page

    with get_db() as conn:
        # total count for this genre
        total_count = conn.execute('''
            SELECT COUNT(*) FROM movies 
            WHERE genres LIKE ?
        ''', (f'%{genre_name}%',)).fetchone()[0] or 0

        total_pages = max(1, math.ceil(total_count / per_page))

        # clamp page to available range and recompute offset
        if page > total_pages:
            page = total_pages
            offset = (page - 1) * per_page

        movies = conn.execute('''
            SELECT * FROM movies 
            WHERE genres LIKE ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        ''', (f'%{genre_name}%', per_page, offset)).fetchall()

    rendered = render_template('genre.html', 
                           genre=genre_name, 
                           movies=[dict(movie) for movie in movies],
                           page=page,
                           total_pages=total_pages,
                           per_page=per_page)
    resp = make_response(rendered)
    resp.headers.setdefault('Cache-Control', 'public, max-age=120, stale-while-revalidate=60')
    return resp

@app.route('/categories')
@rate_limit
@cache.cached(timeout=600)
def categories():
    with get_db() as conn:
        genre_counts = {}
        for genre in app.config['ALL_GENRES']:
            count = conn.execute('''
                SELECT COUNT(*) FROM movies 
                WHERE genres LIKE ?
            ''', (f'%{genre}%',)).fetchone()[0]
            if count > 0:
                genre_counts[genre] = count
        
        genre_previews = {}
        for genre in genre_counts.keys():
            movies = conn.execute('''
                SELECT id, title, image FROM movies 
                WHERE genres LIKE ? 
                ORDER BY RANDOM() 
                LIMIT 3
            ''', (f'%{genre}%',)).fetchall()
            genre_previews[genre] = [dict(movie) for movie in movies]
    
    resp = make_response(render_template('categories.html', 
                         genre_counts=genre_counts,
                         genre_previews=genre_previews))
    resp.headers.setdefault('Cache-Control', 'public, max-age=600, stale-while-revalidate=60')
    return resp

@app.route('/admin/analytics')
@rate_limit
def analytics():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))

    with get_analytics_db() as conn:
        total_visits = conn.execute('SELECT COUNT(*) FROM page_views').fetchone()[0] or 0
        unique_visitors = conn.execute('SELECT COUNT(*) FROM unique_visitors').fetchone()[0] or 0
        watch_now_clicks = conn.execute('SELECT COUNT(*) FROM button_clicks WHERE button_type = "watch"').fetchone()[0] or 0
        conversion_rate = round((watch_now_clicks / total_visits * 100), 2) if total_visits > 0 else 0

        thirty_days_ago = (get_kolkata_time() - timedelta(days=30)).strftime('%Y-%m-%d')
        daily_traffic = conn.execute('''
            SELECT date, visits, unique_visitors, returning_visitors
            FROM daily_traffic 
            WHERE date >= ? 
            ORDER by date ASC
        ''', (thirty_days_ago,)).fetchall()

        daily_watch_clicks = conn.execute('''
            SELECT date(timestamp) as date, COUNT(*) as count
            FROM button_clicks
            WHERE button_type = 'watch' AND date(timestamp) >= ?
            GROUP BY date(timestamp)
            ORDER BY date(timestamp) ASC
        ''', (thirty_days_ago,)).fetchall()

        watch_clicks_dict = {row['date']: row['count'] for row in daily_watch_clicks}

        formatted_daily_traffic = []
        for day in daily_traffic:
            day_dict = dict(day)
            date_obj = datetime.strptime(day['date'], '%Y-%m-%d')
            day_dict['formatted_date'] = date_obj.strftime('%b %d')
            day_dict['watch_now_clicks'] = watch_clicks_dict.get(day['date'], 0)
            formatted_daily_traffic.append(day_dict)

        device_distribution = conn.execute('''
            SELECT device_type, device_model, COUNT(*) as count 
            FROM page_views 
            GROUP BY device_type, device_model
            ORDER BY count DESC
        ''').fetchall()

        browser_distribution = conn.execute('''
            SELECT browser, os, COUNT(*) as count 
            FROM page_views 
            GROUP BY browser, os
            ORDER BY count DESC
        ''').fetchall()

        recent_visits = conn.execute('''
            SELECT 
                datetime(timestamp) as timestamp,
                page_url,
                device_type,
                device_model,
                browser,
                os,
                button_clicked
            FROM page_views 
            WHERE page_url NOT LIKE '/admin%'
            ORDER BY timestamp DESC 
            LIMIT 20
        ''').fetchall()

        most_watched = conn.execute('''
            SELECT movie_id, COUNT(*) as watch_count 
            FROM button_clicks 
            WHERE button_type = 'watch' AND movie_id IS NOT NULL
            GROUP BY movie_id 
            ORDER BY watch_count DESC 
            LIMIT 10
        ''').fetchall()

        movie_titles = {}
        if most_watched:
            with get_db() as movie_conn:
                for item in most_watched:
                    movie = movie_conn.execute('SELECT title FROM movies WHERE id = ?', (item['movie_id'],)).fetchone()
                    if movie:
                        movie_titles[item['movie_id']] = movie['title']

        device_models = conn.execute('''
            SELECT device_model, COUNT(*) as count 
            FROM page_views 
            WHERE device_model != 'Unknown'
            GROUP BY device_model 
            ORDER BY count DESC 
            LIMIT 10
        ''').fetchall()

    return render_template(
        'analytics.html',
        total_visits=total_visits,
        unique_visitors=unique_visitors,
        watch_now_clicks=watch_now_clicks,
        conversion_rate=conversion_rate,
        daily_traffic=formatted_daily_traffic,
        device_distribution=device_distribution,
        browser_distribution=browser_distribution,
        recent_visits=recent_visits,
        most_watched=most_watched,
        movie_titles=movie_titles,
        device_models=device_models
    )

@app.context_processor
def inject_media_helpers():
    def media_url(value, folder=None, fallback=None):
        if not value:
            return fallback
        value_str = str(value)
        if value_str.startswith(('http://', 'https://')):
            return value_str
        if folder:
            return url_for('static', filename=f'{folder}/{value_str}')
        return url_for('static', filename=value_str)
    # Add guess_video_mime to template context
    return {
        'media_url': media_url,
        'guess_video_mime': guess_video_mime
    }

@app.route('/upload', methods=['POST'])
def upload_original():
    """Upload any large file directly (no transcoding)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Invalid filename'}), 400

    filename = secure_filename(file.filename)
    save_dir = '/var/www/files'
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    try:
        with open(path, 'wb') as f:
            shutil.copyfileobj(file.stream, f)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    return jsonify({
        'message': 'Upload successful ✅',
        'url': f'http://217.217.249.3/files/{filename}'
    })

# ---------------------------
# API helpers & endpoints
# ---------------------------

def require_api_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-API-KEY")
        if not token or token != app.config.get('API_TOKEN'):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

def movie_row_to_dict(row):
    if not row:
        return None
    movie = dict(row)
    # Provide links as list and image as path (keep filename in DB but return full static path)
    links_list = [l.strip() for l in movie.get('links', '').split(',')] if movie.get('links') else []
    image_value = movie.get('image') if movie.get('image') else None
    if image_value and str(image_value).startswith('http'):
        image_url = image_value
    elif image_value:
        image_url = f"/static/images/{image_value}"
    else:
        image_url = None
    movie['links'] = links_list
    movie['image'] = image_url
    return movie

def save_image_file(file_storage):
    """Upload image to remote storage and return public URL (or None)."""
    if not file_storage or not file_storage.filename:
        return None
    if not allowed_image_file(file_storage.filename):
        return None
    return upload_to_remote_storage(file_storage)

def delete_image_file(filename):
    """Delete image filename from upload folder if exists"""
    if not filename:
        return
    if str(filename).startswith('http'):
        return
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        # don't crash on delete failure
        print(f"Failed to delete image {path}: {e}")

# Create movie (multipart/form-data)
@app.route('/api/movies', methods=['POST'])
@rate_limit
@require_api_token
def api_create_movie():
    # Accept both form-data and JSON; for file upload, use form-data
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        title = request.form.get('title', '').strip()
        genres = request.form.get('genres', '').strip()
        links = request.form.getlist('links') or []
        # also allow a single 'links' comma-separated string
        if not links and request.form.get('links'):
            links = [l.strip() for l in request.form.get('links').split(',') if l.strip()]
        image_file = request.files.get('image')
    else:
        data = request.get_json() or {}
        title = (data.get('title') or '').strip()
        genres = (data.get('genres') or '').strip()
        links = data.get('links') or []
        image_file = None  # can't upload file via JSON

    if not title:
        return jsonify({"error": "Title is required"}), 400

    links_str = ','.join([l.strip() for l in links if l.strip()])

    image_filename = None
    if image_file:
        if not allowed_image_file(image_file.filename):
            return jsonify({"error": "Invalid image file type"}), 400
        image_filename = save_image_file(image_file)

    created_at = get_kolkata_time().strftime('%Y-%m-%d %H:%M:%S')

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO movies (title, image, links, genres, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, image_filename or '', links_str or '', genres or '', created_at))
        movie_id = cursor.lastrowid
        conn.commit()

    with get_db() as conn:
        movie = conn.execute('SELECT * FROM movies WHERE id = ?', (movie_id,)).fetchone()
    return jsonify({"message": "Movie created", "movie": movie_row_to_dict(movie)}), 201

# Get all movies
@app.route('/api/movies', methods=['GET'])
@rate_limit
@require_api_token
def api_get_movies():
    with get_db() as conn:
        movies = conn.execute('SELECT * FROM movies ORDER BY created_at DESC').fetchall()
        result = [movie_row_to_dict(m) for m in movies]
    return jsonify(result)

# Get single movie
@app.route('/api/movies/<int:movie_id>', methods=['GET'])
@rate_limit
@require_api_token
def api_get_movie(movie_id):
    with get_db() as conn:
        movie = conn.execute('SELECT * FROM movies WHERE id = ?', (movie_id,)).fetchone()
        if not movie:
            return jsonify({"error": "Movie not found"}), 404
        return jsonify(movie_row_to_dict(movie))

# Update movie (supports multipart/form-data or JSON)
@app.route('/api/movies/<int:movie_id>', methods=['PUT'])
@rate_limit
@require_api_token
def api_update_movie(movie_id):
    # Fetch existing movie
    with get_db() as conn:
        existing = conn.execute('SELECT * FROM movies WHERE id = ?', (movie_id,)).fetchone()
        if not existing:
            return jsonify({"error": "Movie not found"}), 404

    # Prepare update values
    image_filename = None
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        title = request.form.get('title', existing['title']).strip()
        genres = request.form.get('genres', existing['genres']).strip()
        links = request.form.getlist('links') or []
        if not links and request.form.get('links'):
            links = [l.strip() for l in request.form.get('links').split(',') if l.strip()]
        image_file = request.files.get('image')
        if image_file and image_file.filename:
            if not allowed_image_file(image_file.filename):
                return jsonify({"error": "Invalid image file type"}), 400
            image_filename = save_image_file(image_file)
    else:
        data = request.get_json() or {}
        title = (data.get('title') or existing['title']).strip()
        genres = (data.get('genres') or existing['genres']).strip()
        links = data.get('links') if 'links' in data else ([l.strip() for l in existing['links'].split(',')] if existing['links'] else [])
        image_filename = None

    links_str = ','.join([l.strip() for l in links if l.strip()])

    # If a new image uploaded, delete the old image file
    if image_filename:
        old_image = existing['image'] if existing['image'] else None
        if old_image and old_image != image_filename:
            delete_image_file(old_image)

    # Create SQL update
    with get_db() as conn:
        cursor = conn.cursor()
        if image_filename:
            cursor.execute('''
                UPDATE movies SET title = ?, genres = ?, links = ?, image = ? WHERE id = ?
            ''', (title, genres, links_str, image_filename, movie_id))
        else:
            cursor.execute('''
                UPDATE movies SET title = ?, genres = ?, links = ? WHERE id = ?
            ''', (title, genres, links_str, movie_id))
        conn.commit()

        movie = conn.execute('SELECT * FROM movies WHERE id = ?', (movie_id,)).fetchone()

    return jsonify({"message": "Movie updated", "movie": movie_row_to_dict(movie)})

# Delete movie (and delete image file)
@app.route('/api/movies/<int:movie_id>', methods=['DELETE'])
@rate_limit
@require_api_token
def api_delete_movie(movie_id):
    with get_db() as conn:
        movie = conn.execute('SELECT * FROM movies WHERE id = ?', (movie_id,)).fetchone()
        if not movie:
            return jsonify({"error": "Movie not found"}), 404
        # delete image file if it exists
        image_filename = movie['image'] if movie['image'] else None
        if image_filename:
            delete_image_file(image_filename)
        conn.execute('DELETE FROM movies WHERE id = ?', (movie_id,))
        conn.commit()
    return jsonify({"message": "Movie deleted"})

@app.route('/api/active_users')
@rate_limit
@require_api_token
def api_active_users():
    """Return count of active users (sessions with last_seen within last 5 minutes)."""
    with get_analytics_db() as conn:
        cutoff = (datetime.utcnow() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
        row = conn.execute('SELECT COUNT(*) as cnt FROM active_sessions WHERE last_seen >= ?', (cutoff,)).fetchone()
        count = row['cnt'] if row else 0
    return jsonify({"active_users": count})

@app.route('/api/watch_count/<int:movie_id>')
@rate_limit
def api_watch_count(movie_id):
    with get_analytics_db() as conn:
        row = conn.execute(
            'SELECT COUNT(*) as count FROM button_clicks WHERE button_type = "watch" AND movie_id = ?',
            (movie_id,)
        ).fetchone()
        count = row['count'] if row else 0
    return jsonify({"count": count})

@app.route('/api/log_play/<int:movie_id>', methods=['POST'])
@rate_limit
def log_play(movie_id):
    """Logs a play event for a movie."""
    try:
        with get_analytics_db() as conn:
            conn.execute('''
                INSERT INTO button_clicks (page_url, button_type, movie_id, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (f'/movie/{movie_id}', 'watch', movie_id, get_kolkata_time().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
        return jsonify({"success": True, "message": "Play logged"}), 200
    except Exception as e:
        app.logger.error(f"Error logging play for movie {movie_id}: {e}")
        return jsonify({"success": False, "message": "Internal server error"}), 500

@app.route('/09d984175eab93c1604825515cdc4e37.html')
def exoclick_verification():
    return render_template('09d984175eab93c1604825515cdc4e37.html')


@app.route('/robots.txt')
def robots_txt():
    """Serve robots.txt from the static folder."""
    # Use send_from_directory to ensure proper headers and path handling
    static_folder = app.static_folder or 'static'
    return send_from_directory(static_folder, 'robots.txt')

# Add import-time initialization so tables exist under WSGI / import
init_db()
init_analytics_db()

init_db()
init_analytics_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
