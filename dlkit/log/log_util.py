import logging

def logging_set_up(log_file_basename,
                   log_file_format    = '%(asctime)s - %(name)s - %(levelname)-8s %(message)s',
                   log_console_level  = logging.WARNING,
                   log_console_format = '%(name)s - %(levelname)-8s %(message)s'):
    """Sets up loggers."""
    # set up logging of debug to file
    # Note: The level for `basicConfig` has to be lower than all other handlers added afterwards.
    logging.basicConfig(level=logging.DEBUG,
                        format=log_file_format,
                        datefmt='%y-%m-%d %H:%M',
                        filename=str(log_file_basename)+'_debug.log',
                        filemode='w')

    # set up formatter
    formatter = logging.Formatter(log_file_format)

    # set up logging of info to file
    handler = logging.FileHandler(str(log_file_basename)+'_info.log', mode='w')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(handler)

    # set up logging of errors to file
    handler = logging.FileHandler(str(log_file_basename)+'_error.log', mode='w')
    handler.setFormatter(formatter)
    handler.setLevel(logging.ERROR)
    logging.getLogger('').addHandler(handler)

    # set up logging to console
    if log_console_level is not None:
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(log_console_level)
        # set a format for console use
        formatter = logging.Formatter(log_console_format)
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

def logging_get_logger(name=''):
    """Gets a logger by its name. Returns root logger by default."""
    return logging.getLogger(name)
