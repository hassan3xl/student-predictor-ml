from django.conf import settings
from django.core.management.base import BaseCommand

from predictor import training


class Command(BaseCommand):
    help = "Train the at-risk models offline and save them to ML_ARTIFACT_DIR."

    def add_arguments(self, parser):
        parser.add_argument("--data", default=str(settings.SAMPLE_DATA),
                            help="CSV with the dataset columns plus G3 (final grade).")

    def handle(self, *args, **opts):
        img_dir = settings.BASE_DIR / "predictor" / "static" / "predictor" / "img"
        training.train(opts["data"], settings.ML_ARTIFACT_DIR, img_dir, log=self.stdout.write)
        self.stdout.write(self.style.SUCCESS("Training complete."))
