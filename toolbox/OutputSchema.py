from pathlib import Path

from toolbox.Log import Log


class OutputPath:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.dir_path_visualize = output_path / 'visualize'
        self.dir_path_visualize.mkdir(parents=True, exist_ok=True)
        self.dir_path_embedding = output_path / 'embedding'
        self.dir_path_embedding.mkdir(parents=True, exist_ok=True)
        self.dir_path_log = output_path / 'log'
        self.dir_path_log.mkdir(parents=True, exist_ok=True)
        self.dir_path_checkpoint = output_path / 'checkpoint'
        self.dir_path_checkpoint.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, filename="checkpoint.tar") -> Path:
        return self.dir_path_checkpoint / filename

    def log_path(self, filename) -> Path:
        return self.dir_path_log / filename

    def embedding_path(self, filename) -> Path:
        return self.dir_path_embedding / filename

    def entity_embedding_path(self, score=-1) -> Path:
        if score == -1:
            return self.embedding_path("entity_embedding.txt")
        else:
            return self.embedding_path("entity_embedding_score_%d.txt" % int(score))

    def visualize_path(self, filename) -> Path:
        return self.dir_path_visualize / filename


class OutputSchema:
    """./output
        - experiment name
          - visualize
            - events...
          - log
            - train.log
            - test.log
            - valid.log
          - checkpoint
            - checkpoint_score_xx.tar
          - embedding
            - embedding_score_xx.pkl
          - config.yaml

        Args:
            experiment_name (str): Name of this experiment

        Examples:
            >>> from toolbox.OutputSchema import OutputSchema
            >>> kgdata = OutputSchema("dL50a_TransE")
            >>> kgdata.dump()

    """

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.root_path = self.output_home_path()
        self.output_path = OutputPath(self.root_path)
        self.logger = Log(str(self.root_path / "output.log"), name_scope="output")

    def output_home_path(self) -> Path:
        data_home_path: Path = Path('.') / 'output'
        data_home_path.mkdir(parents=True, exist_ok=True)
        data_home_path = data_home_path.resolve()
        return data_home_path / self.name

    def output_path_child(self, name) -> Path:
        return self.root_path / name

    def dump(self):
        """ Displays all the metadata of the knowledge graph"""
        for key, value in self.__dict__.items():
            self.logger.info("%s %s" % (key, value))
