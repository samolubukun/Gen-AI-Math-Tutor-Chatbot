# AI Math Tutor Pro

An advanced AI-powered math tutor web app built with **FastAPI**, leveraging **Gemini 1.5 Flash**, **MongoDB**, and **JWT-based authentication**. Supports text and image-based math questions with graph generation and step-by-step solutions.

## 🔧 Features

- ✍️ Ask math questions (text or image-based)
- 📊 Automatic graph plotting for functions
- 🤖 AI-generated step-by-step math solutions (via Gemini Flash)
- 🖼️ Image input support (e.g., handwritten math problems)
- 🔒 User authentication (register/login/logout)
- 🧠 Chat history (view past questions and answers)
- 📁 Clean HTML frontend with Jinja2 and static assets

## ⚙️ Tech Stack

- **Backend:** FastAPI
- **AI Model:** Google Gemini 1.5 Flash
- **Database:** MongoDB (via Motor)
- **Security:** JWT, bcrypt
- **Frontend:** Jinja2 Templates, HTML/CSS
- **Others:** Python, Matplotlib, Pillow


## 📸 Screenshots
![png (2)](https://github.com/user-attachments/assets/355cb6c3-b50d-48f9-ba1e-70ffa131751e)
![png (3)](https://github.com/user-attachments/assets/cb1ec043-c534-4c56-b678-af058934788f)

## 🌐 Endpoints Overview

- `GET /` – Homepage (Login/Register)
- `POST /register` – Create new user
- `POST /login` – Authenticate user
- `GET /dashboard` – User dashboard
- `POST /ask` – Submit math question (text/image/graph)
- `GET /history` – View question history
- `GET /logout` – Logout user

---

**.env Variables Used:**

```env
SECRET_KEY=your_secret_key
GOOGLE_AI_API_KEY=your_google_gemini_api_key
MONGODB_URL=your_mongodb_connection_string
