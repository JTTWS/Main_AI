print("Bot testi basliyor...")
print("=" * 50)

import sys
sys.path.append('/Users/serkanozturk/Desktop/JTTWS')

try:
    from bot_config import BotConfig
    print("Config yuklendi")
    print(f"Sermaye: ${BotConfig.INITIAL_CAPITAL}")
    print(f"Ciftler: {BotConfig.PAIRS}")
    print("Bot ayarlari dogru!")
except Exception as e:
    print(f"Hata: {e}")

print("=" * 50)
print("Test bitti")
