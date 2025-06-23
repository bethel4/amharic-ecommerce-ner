#!/usr/bin/env python3
"""
Telegram E-commerce Channel Scraper for Amharic Data Collection

This script scrapes messages from Ethiopian e-commerce Telegram channels
to collect data for NER training and analysis.
"""

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import pandas as pd
from telethon import TelegramClient, events
from telethon.tl.types import Message, DocumentAttributeFilename
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TelegramScraper:
    """Telegram scraper for collecting e-commerce messages from Ethiopian channels."""
    
    def __init__(self, api_id: str, api_hash: str, phone: str):
        """
        Initialize the Telegram scraper.
        
        Args:
            api_id: Telegram API ID
            api_hash: Telegram API Hash
            phone: Phone number for authentication
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = None
        self.db_path = "data/raw/telegram_data.db"
        
        # Ensure data directory exists
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Ethiopian e-commerce channels to scrape
        self.channels = {
            "mertteka": "@MerttEka",
            "forfreemarket": "@forfreemarket",
            "classybrands": "@classybrands",
            "marakibrand": "@marakibrand",
            "aradabrand2": "@aradabrand2",
            "marakisat2": "@marakisat2",
            "belaclassic": "@belaclassic",
            "awasmart": "@AwasMart",
            "qnashcom": "@qnashcom"
        }
        
    def _init_database(self):
        """Initialize SQLite database for storing scraped data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                channel_name TEXT,
                channel_id INTEGER,
                sender_id INTEGER,
                sender_name TEXT,
                message_text TEXT,
                message_date TIMESTAMP,
                is_forwarded BOOLEAN,
                has_media BOOLEAN,
                media_type TEXT,
                media_path TEXT,
                reply_to_message_id INTEGER,
                views INTEGER,
                forwards INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create channels table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_name TEXT UNIQUE,
                channel_id INTEGER,
                title TEXT,
                description TEXT,
                member_count INTEGER,
                last_scraped TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    async def start_client(self):
        """Start the Telegram client."""
        try:
            self.client = TelegramClient('session_name', self.api_id, self.api_hash)
            await self.client.start(phone=self.phone)
            logger.info("Telegram client started successfully")
        except Exception as e:
            logger.error(f"Failed to start Telegram client: {e}")
            raise
    
    async def get_channel_info(self, channel_username: str) -> Dict:
        """Get information about a Telegram channel."""
        try:
            entity = await self.client.get_entity(channel_username)
            return {
                'id': entity.id,
                'title': entity.title,
                'username': entity.username,
                'description': getattr(entity, 'about', ''),
                'member_count': getattr(entity, 'participants_count', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get info for {channel_username}: {e}")
            return {}
    
    async def download_media(self, message: Message, channel_name: str) -> Optional[str]:
        """Download media from a message."""
        if not message.media:
            return None
            
        try:
            # Create media directory
            media_dir = Path(f"data/raw/media/{channel_name}")
            media_dir.mkdir(parents=True, exist_ok=True)
            
            # Download media
            filename = f"{message.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            path = await message.download_media(file=f"{media_dir}/{filename}")
            
            return str(path) if path else None
            
        except Exception as e:
            logger.error(f"Failed to download media from message {message.id}: {e}")
            return None
    
    def extract_message_data(self, message: Message, channel_name: str) -> Dict:
        """Extract relevant data from a Telegram message."""
        # Determine media type
        media_type = None
        if message.media:
            if hasattr(message.media, 'photo'):
                media_type = 'photo'
            elif hasattr(message.media, 'document'):
                media_type = 'document'
            elif hasattr(message.media, 'webpage'):
                media_type = 'webpage'
        
        # Safely get sender name (user or channel)
        sender_name = None
        if message.sender:
            sender_name = getattr(message.sender, 'first_name', None) or getattr(message.sender, 'title', None)
        
        return {
            'message_id': message.id,
            'channel_name': channel_name,
            'channel_id': message.peer_id.channel_id if hasattr(message.peer_id, 'channel_id') else None,
            'sender_id': message.sender_id,
            'sender_name': sender_name,
            'message_text': message.text or message.raw_text or '',
            'message_date': message.date.isoformat(),
            'is_forwarded': message.forward is not None,
            'has_media': message.media is not None,
            'media_type': media_type,
            'reply_to_message_id': message.reply_to.reply_to_msg_id if message.reply_to else None,
            'views': getattr(message, 'views', 0),
            'forwards': getattr(message, 'forwards', 0)
        }
    
    async def scrape_channel(self, channel_name: str, limit: int = 1000) -> List[Dict]:
        """Scrape messages from a specific channel."""
        messages_data = []
        
        try:
            entity = await self.client.get_entity(channel_name)
            logger.info(f"Starting to scrape {channel_name} (limit: {limit})")
            
            async for message in self.client.iter_messages(entity, limit=limit):
                if message.text or message.media:  # Only process messages with content
                    # Extract message data
                    message_data = self.extract_message_data(message, channel_name)
                    
                    # Download media if present
                    if message.media:
                        media_path = await self.download_media(message, channel_name)
                        message_data['media_path'] = media_path
                    
                    messages_data.append(message_data)
                    
                    # Log progress every 100 messages
                    if len(messages_data) % 100 == 0:
                        logger.info(f"Scraped {len(messages_data)} messages from {channel_name}")
            
            logger.info(f"Successfully scraped {len(messages_data)} messages from {channel_name}")
            
        except Exception as e:
            logger.error(f"Failed to scrape {channel_name}: {e}")
        
        return messages_data
    
    def save_to_database(self, messages_data: List[Dict], channel_name: str):
        """Save scraped messages to SQLite database."""
        if not messages_data:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for message_data in messages_data:
                cursor.execute('''
                    INSERT OR REPLACE INTO messages (
                        message_id, channel_name, channel_id, sender_id, sender_name,
                        message_text, message_date, is_forwarded, has_media, media_type,
                        media_path, reply_to_message_id, views, forwards
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message_data['message_id'],
                    message_data['channel_name'],
                    message_data['channel_id'],
                    message_data['sender_id'],
                    message_data['sender_name'],
                    message_data['message_text'],
                    message_data['message_date'],
                    message_data['is_forwarded'],
                    message_data['has_media'],
                    message_data['media_type'],
                    message_data['media_path'],
                    message_data['reply_to_message_id'],
                    message_data['views'],
                    message_data['forwards']
                ))
            
            # Update channel info
            cursor.execute('''
                INSERT OR REPLACE INTO channels (
                    channel_name, last_scraped
                ) VALUES (?, ?)
            ''', (channel_name, datetime.now().isoformat()))
            
            conn.commit()
            logger.info(f"Saved {len(messages_data)} messages to database for {channel_name}")
            
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def export_to_csv(self, channel_name: str = None):
        """Export scraped data to CSV format."""
        conn = sqlite3.connect(self.db_path)
        
        if channel_name:
            query = "SELECT * FROM messages WHERE channel_name = ?"
            df = pd.read_sql_query(query, conn, params=[channel_name])
            filename = f"data/raw/{channel_name}_messages.csv"
        else:
            df = pd.read_sql_query("SELECT * FROM messages", conn)
            filename = "data/raw/all_messages.csv"
        
        df.to_csv(filename, index=False, encoding='utf-8')
        conn.close()
        
        logger.info(f"Exported data to {filename}")
        return filename
    
    async def scrape_all_channels(self, limit_per_channel: int = 1000):
        """Scrape messages from all configured channels."""
        if not self.client:
            await self.start_client()
        
        total_messages = 0
        
        for channel_key, channel_username in self.channels.items():
            logger.info(f"Processing channel: {channel_key} ({channel_username})")
            
            try:
                # Scrape messages
                messages_data = await self.scrape_channel(channel_username, limit_per_channel)
                
                if messages_data:
                    # Save to database
                    self.save_to_database(messages_data, channel_key)
                    total_messages += len(messages_data)
                    
                    # Export to CSV
                    self.export_to_csv(channel_key)
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing {channel_key}: {e}")
                continue
        
        logger.info(f"Scraping completed. Total messages collected: {total_messages}")
        
        # Export all data
        self.export_to_csv()
    
    async def close(self):
        """Close the Telegram client."""
        if self.client:
            await self.client.disconnect()
            logger.info("Telegram client disconnected")


async def main():
    """Main function to run the scraper."""
    # Get credentials from environment variables
    api_id = os.getenv('TELEGRAM_API_ID')
    api_hash = os.getenv('TELEGRAM_API_HASH')
    phone = os.getenv('TELEGRAM_PHONE')
    
    if not all([api_id, api_hash, phone]):
        logger.error("Missing Telegram credentials. Please set TELEGRAM_API_ID, TELEGRAM_API_HASH, and TELEGRAM_PHONE environment variables.")
        return
    
    scraper = TelegramScraper(api_id, api_hash, phone)
    
    try:
        await scraper.scrape_all_channels(limit_per_channel=500)  # Start with 500 messages per channel
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
