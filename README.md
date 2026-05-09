# CollegeSync

AI-powered notice board for colleges. Students ask questions in plain English 
and get instant, verified answers from admin-uploaded notices and documents.

## Why not just a WhatsApp group?
- Semantic search across hundreds of documents
- Admin-verified information only — no misinformation
- Available 24/7 with instant answers, not manual scrolling

## Architecture
FastAPI backend → ChromaDB (vector search) → Gemini (answer generation)
Firebase Auth (RBAC) → Redis (caching) → S3 (file storage)

## Running locally
1. Clone the repo
2. Copy .env.example and fill in keys
3. docker build -t collegesync . && docker run -p 8000:8000 collegesync
