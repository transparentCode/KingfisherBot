import os
import csv
import pandas as pd
from flask import Blueprint, jsonify, request, current_app

asset_info_bp = Blueprint('asset_info', __name__)

# Cache for the asset data to avoid reading the CSV repeatedly
_assets_cache = None
_last_modified = 0


def get_assets_data():
    """Load assets data from CSV and cache it"""
    global _assets_cache, _last_modified

    csv_path = os.path.join(current_app.root_path, 'indices', 'ind_nifty100list.csv')

    # Check if file has been modified since last read
    current_modified = os.path.getmtime(csv_path)
    if _assets_cache is None or current_modified > _last_modified:
        df = pd.read_csv(csv_path)
        _assets_cache = df.to_dict(orient='records')
        _last_modified = current_modified

    return _assets_cache


@asset_info_bp.route('/api/assets', methods=['GET'])
def get_all_assets():
    """Get all assets with optional filtering"""
    assets = get_assets_data()

    # Optional filtering
    industry = request.args.get('industry')
    if industry:
        assets = [asset for asset in assets if asset['Industry'] == industry]

    return jsonify({
        'count': len(assets),
        'assets': assets
    })


@asset_info_bp.route('/api/assets/<symbol>', methods=['GET'])
def get_asset_by_symbol(symbol):
    """Get asset details by symbol"""
    assets = get_assets_data()

    # Find asset by symbol (case insensitive)
    for asset in assets:
        if asset['Symbol'].lower() == symbol.lower():
            return jsonify(asset)

    return jsonify({'error': 'Asset not found'}), 404


@asset_info_bp.route('/api/assets/industries', methods=['GET'])
def get_industries():
    """Get list of unique industries"""
    assets = get_assets_data()
    industries = sorted(list(set(asset['Industry'] for asset in assets)))

    return jsonify({
        'count': len(industries),
        'industries': industries
    })


@asset_info_bp.route('/api/assets/search', methods=['GET'])
def search_assets():
    """Search assets by name or symbol"""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'error': 'Search query parameter (q) is required'}), 400

    assets = get_assets_data()
    results = [
        asset for asset in assets
        if query in asset['Company Name'].lower() or
           query in asset['Symbol'].lower()
    ]

    return jsonify({
        'count': len(results),
        'results': results
    })