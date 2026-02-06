# image signatures
signatures = {
        'jpeg': (
                b'\xff\xd8\xff\xdb',  # JPEG raw
                b'\xff\xd8\xff\xe0',  # JPEG/JFIF
                b'\xff\xd8\xff\xe1',  # JPEG/EXIF
                b'\xff\xd8\xff\xe2',  # JPEG Canon
                b'\xff\xd8\xff\xe3',  # Samsung
                b'\xff\xd8\xff\xe8',  # JPEG/SPIFF
                b'\xff\xd8\xff\xee',  # Adobe JPEG
                ),
        'png': (b'\x89PNG\r\n\x1a\n',),
        'gif': (b'GIF87a', b'GIF89a'),
        'bmp': (b'BM',),
        'tiff': (
            b'II\x2a\x00',  # Little-endian
            b'MM\x00\x2a',  # Big-endian
        ),
    }
# file length limit
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
