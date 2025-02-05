# Google Gemini Image Library Agent
#
# Author: Peter Jakubowski
# Date: 2/1/2025
# Description: Database configuration
#

import fastlite
from schemas import Assets, AssetMetadata, GenerativeMetadata, Embeddings

# =========================
# ======= Database ========
# =========================

db = fastlite.database('image_library.db')

# Create core table
assets = db.t.assets
if not assets.exists():
    assets = db.create(Assets, pk="id")

# Create metadata table
asset_metadata = db.t.asset_metadata
if not asset_metadata.exists():
    asset_metadata = db.create(AssetMetadata, pk="id")

# Create generative table
generative_metadata = db.t.generative_metadata
if not generative_metadata.exists():
    generative_metadata = db.create(GenerativeMetadata, pk="id")

# Create embedding table
embeddings = db.t.embeddings
if not embeddings.exists():
    embeddings = db.create(Embeddings)
