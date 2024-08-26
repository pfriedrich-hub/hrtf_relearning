import freefield

def init_proc(model, connection, led, buttons):
    self.processor = processor
    if self.processor == 'RP2':
        self.zbus = True
        self.connection = 'zBus'
        self.led_feedback = False
        self.button_control = True
    elif self.processor == 'RM1':
        self.zbus = False
        self.connection = 'USB'
        self.led_feedback = False
        self.button_control = False