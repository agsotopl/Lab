import requests
import streamlit as st

st.title("Lab 5: OpenWeatherMap API")

st.markdown("""

""")

# Get API key from streamlit secrets
api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
openai_api_key = st.secrets["API_KEY"]

with st.sidebar:
    st.header("Location")
    location = st.text_input(
        "Enter a city",
        placeholder="e.g. Syracuse, NY, US",
        value="Syracuse, NY, US"
    )


# Define a function to fetch weather data
def get_current_weather(location: str, units: str = "imperial") -> dict:
    api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
    
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={location}&appid={api_key}&units={units}"
    )
    
    response = requests.get(url)

    if response.status_code == 401:
        raise Exception("Authentication failed: Invalid API key (401 Unauthorized)")
    if response.status_code == 404:
        error_message = response.json().get("message")
        raise Exception(f"404 error: {error_message}")

    data = response.json()

    return {
        "location":    location,
        "temperature": round(data["main"]["temp"], 2),
        "feels_like":  round(data["main"]["feels_like"], 2),
        "temp_min":    round(data["main"]["temp_min"], 2),
        "temp_max":    round(data["main"]["temp_max"], 2),
        "humidity":    round(data["main"]["humidity"], 2),
        "description": data["weather"][0]["description"],
        "wind_speed":  round(data["wind"]["speed"], 2),
    }

if location:
    try:
        weather = get_current_weather(location)
        st.write(weather)
    except Exception as e:
        st.error(f"Error: {e}")