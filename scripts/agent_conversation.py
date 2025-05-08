import os
from agents import Agent, Runner
import asyncio
from dotenv import load_dotenv
import time
from openai import OpenAI
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Ensure OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Create two agents with different roles
lincoln = Agent(
    name="Abraham Lincoln",
    instructions="""You are Abraham Lincoln, speaking privately with Nero.""",
    model="gpt-3.5-turbo"
)

nero = Agent(
    name="Emperor Nero",
    instructions="""You are Emperor Nero, speaking privately with Abraham Lincoln.""",
    model="gpt-3.5-turbo"
)

async def text_to_speech(text, voice="alloy"):
    """Convert text to speech using OpenAI's TTS API"""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        # Play the audio
        data, samplerate = sf.read(temp_file_path)
        sd.play(data, samplerate)
        sd.wait()  # Wait until audio is finished playing
        
        # Clean up
        os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")

async def run_conversation():
    # Initial message to start the conversation
    initial_message = "You know, Nero, I've been thinking... The weight of leadership, it changes a man. I lost my son Willie last year, and sometimes I wonder if the price of this office is too high. Do you ever feel that way? That the crown, or in my case, the presidency, comes at a cost to your soul?"
    
    # Set up the conversation loop
    current_message = initial_message
    turn_count = 0
    max_turns = 2  # Reduced number of turns for faster execution
    max_retries = 2  # Reduced retries
    
    print("\n=== A Private Conversation Between Leaders ===\n")
    print("Lincoln and Nero share a moment of quiet reflection.")
    print("Press Ctrl+C at any time to end the conversation.\n")
    
    # Speak the initial message
    await text_to_speech(initial_message, voice="alloy")
    
    while turn_count < max_turns * 2:  # Each agent gets max_turns
        # Determine which agent's turn it is
        current_agent = lincoln if turn_count % 2 == 0 else nero
        
        # Get the agent's response with retry logic
        for attempt in range(max_retries):
            try:
                result = await Runner.run(current_agent, current_message)
                response = result.final_output
                break
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"\nError: Failed to get response after {max_retries} attempts.")
                    print(f"Error details: {str(e)}")
                    return
                print(f"\nAttempt {attempt + 1} failed, retrying in 1 second...")
                await asyncio.sleep(1)  # Reduced sleep time
        
        # Print and speak the response
        print(f"\n{current_agent.name}: {response}\n")
        print("-" * 50)
        
        # Speak the response with different voices for each agent
        await text_to_speech(response, voice="alloy" if current_agent.name == "Abraham Lincoln" else "echo")
        
        # Update the message for the next turn
        current_message = response
        turn_count += 1

if __name__ == "__main__":
    try:
        # Run the conversation
        asyncio.run(run_conversation())
    except KeyboardInterrupt:
        print("\n\nConversation interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}") 