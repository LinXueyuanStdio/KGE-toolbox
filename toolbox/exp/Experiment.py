from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Store import StoreSchema
from toolbox.utils.Visualize import VisualizeSchema


class Experiment:

    def __init__(self, output: OutputSchema):
        self.output = output
        self.debug = output.logger.debug
        self.log = output.logger.info
        self.warn = output.logger.warn
        self.error = output.logger.error
        self.critical = output.logger.critical
        self.success = output.logger.success
        self.fail = output.logger.failed
        self.vis = VisualizeSchema(str(output.pathSchema.dir_path_visualize))
        self.store = StoreSchema(output.pathSchema)

    def re_init(self, output: OutputSchema):
        self.output = output
        self.debug = output.logger.debug
        self.log = output.logger.info
        self.warn = output.logger.warn
        self.error = output.logger.error
        self.critical = output.logger.critical
        self.success = output.logger.success
        self.fail = output.logger.failed
        self.vis = VisualizeSchema(str(output.pathSchema.dir_path_visualize))
        self.store = StoreSchema(output.pathSchema)

