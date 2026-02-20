from openai import OpenAI
import requests
import streamlit as st
import json

st.title("Lab 5: Clothing Recommendation Based on Weather")

st.markdown("""
    Tell us a city you're in, interested in going to, or just curious about, and it'll tell you what to wear based on the current weather. 
""")

# Get API keys from streamlit secrets
api_key = st.secrets["OPENWEATHERMAP_API_KEY"]
openai_api_key = st.secrets["API_KEY"]

client = OpenAI(api_key=openai_api_key)

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


# Define the tool for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name in the format 'City, State, Country' e.g. 'Syracuse, NY, US'. Default to 'Syracuse, NY, US' if no location is provided."
                    }
                },
                "required": ["location"]
            }
        }
    }
]


if location:
    if st.button("Get Advice"):
        try:
            user_message = f"What should I wear today in {location}?"

            # First call: let the model invoke the weather tool
            first_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful fashion advisor."},
                    {"role": "user", "content": user_message}
                ],
                tools=tools,
                tool_choice="auto"
            )

            response_message = first_response.choices[0].message

            # Handle the tool call
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                loc = args.get("location", "Syracuse, NY, US")

                # Call the actual weather function
                weather = get_current_weather(loc)

                weather_summary = (
                    f"Location: {weather['location']}, "
                    f"Temperature: {weather['temperature']}°F, "
                    f"Feels like: {weather['feels_like']}°F, "
                    f"High: {weather['temp_max']}°F, "
                    f"Low: {weather['temp_min']}°F, "
                    f"Conditions: {weather['description']}, "
                    f"Humidity: {weather['humidity']}%, "
                    f"Wind: {weather['wind_speed']} mph"
                )

                # Second call: pass weather data back and get advice
                second_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful fashion advisor."},
                        {"role": "user", "content": user_message},
                        response_message,
                        {"role": "tool", "tool_call_id": tool_call.id, "content": weather_summary},
                    ]
                )

                st.write(second_response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {e}")