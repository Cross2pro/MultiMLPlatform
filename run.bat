start cmd /k "cd ml-prediction-system & npm run dev"
start cmd /k "cd ml-prediction-system-backend-flask & python run.py"
timeout /t 3 & start http://localhost:3000
