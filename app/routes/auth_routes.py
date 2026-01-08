import os
import secrets
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv


load_dotenv()

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/api/login', methods=['POST'])
def login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''

    expected_user = os.getenv('AUTH_USERNAME', 'admin')
    expected_pass = os.getenv('AUTH_PASSWORD', 'admin123')

    if username != expected_user or password != expected_pass:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    token = secrets.token_hex(32)
    return jsonify({'success': True, 'token': token, 'user': {'username': username}})
