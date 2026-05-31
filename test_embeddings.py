import asyncio
import os
import urllib.request, json
from supabase import create_client

from dotenv import load_dotenv
load_dotenv('.env')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
GEMINI_KEYS_RAW = os.getenv('GEMINI_API_KEYS', '')
GEMINI_KEYS = [k.strip() for k in GEMINI_KEYS_RAW.split(',') if k.strip()]

def get_embedding(text):
    key = GEMINI_KEYS[0]
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={key}'
    data = json.dumps({'model': 'models/gemini-embedding-2', 'content': {'parts': [{'text': text}]}, 'outputDimensionality': 768}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    response = urllib.request.urlopen(req)
    result = json.loads(response.read().decode('utf-8'))
    return result['embedding']['values']

async def main():
    print('Testing Telegram (chat_history)...')
    # Use the admin chat id that is in the DB
    chat_id = '8284113566'
    
    # Let's insert a fake message first just to be sure we have something to match!
    emb = get_embedding('Hello world this is a test about grammar')
    supabase.table('chat_history').insert({
        'chat_id': chat_id,
        'role': 'user',
        'content': 'Hello world this is a test about grammar',
        'embedding': emb
    }).execute()
    
    query_emb = get_embedding('grammar test')
    
    res = supabase.rpc('match_user_chat_history', {
        'query_embedding': query_emb,
        'target_user_id': chat_id,
        'history_table': 'chat_history',
        'user_column': 'chat_id',
        'match_threshold': 0.45,
        'match_count': 5
    }).execute()
    
    print('Telegram matches:', len(res.data))
    for m in res.data:
        print(f"  -> [{m['role']}] {m['content']} (Score: {m['similarity']:.3f})")
        
    print('\nTesting WhatsApp (whatsapp_chat_history)...')
    wa_phone = '123456789'
    wa_emb = get_embedding('My favorite color is blue and I love past simple tense')
    supabase.table('whatsapp_chat_history').insert({
        'phone_number': wa_phone,
        'role': 'user',
        'content': 'My favorite color is blue and I love past simple tense',
        'embedding': wa_emb
    }).execute()
    
    wa_query_emb = get_embedding('past simple')
    
    res_wa = supabase.rpc('match_user_chat_history', {
        'query_embedding': wa_query_emb,
        'target_user_id': wa_phone,
        'history_table': 'whatsapp_chat_history',
        'user_column': 'phone_number',
        'match_threshold': 0.45,
        'match_count': 5
    }).execute()
    
    print('WhatsApp matches:', len(res_wa.data))
    for m in res_wa.data:
        print(f"  -> [{m['role']}] {m['content']} (Score: {m['similarity']:.3f})")

if __name__ == '__main__':
    asyncio.run(main())
