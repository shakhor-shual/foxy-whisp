#logic/foxy_config.py
import logging

# Константы
SAMPLING_RATE = 16000
PACKET_SIZE = 2 * SAMPLING_RATE * 5 * 60  # 5 minutes
MOSES = False
WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")
MAX_BUFFER_SECONDS = 15
MAX_PROMPT_SIZE = 180  # Максимальная длина подсказки для Whisper
MAX_INCOMPLETE_SIZE =200 # Максимальная длина непотвержденного фрагмента
MAX_TAIL_SIZE = 500  # Максимальная длина хвоста текста

CONNECTION_SELECT_DELAY = 0.1  # Delay for each select call in seconds
CONNECTION_TOPIC="foxy-whisp"

