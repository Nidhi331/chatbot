To run the server on port 8000, please follow below steps:
cd backend
sudo apt install python3.10-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

To run UI on port 3000:
cd frontend
npm install 
npm start
