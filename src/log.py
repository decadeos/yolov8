import logging
import colorlog

handler = colorlog.StreamHandler()

formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

handler.setFormatter(formatter)

# Настраиваем логгер
logger = colorlog.getLogger('example_logger')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# # Тестируем вывод
# logger.debug('Отладочное сообщение')
# logger.info('Информационное сообщение')
# logger.warning('Предупреждение')
# logger.error('Сообщение об ошибке')
# logger.critical('Критическая ошибка')

