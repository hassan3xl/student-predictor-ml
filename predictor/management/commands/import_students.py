from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from predictor.importer import import_dataframe
from predictor.mlkit import read_table
from predictor.models import Student


class Command(BaseCommand):
    help = "Import students from a CSV (default: the bundled UCI sample) and score them."

    def add_arguments(self, parser):
        parser.add_argument("--file", default=str(settings.SAMPLE_DATA))
        parser.add_argument("--clear", action="store_true", help="Delete existing students first.")

    def handle(self, *args, **opts):
        if opts["clear"]:
            deleted, _ = Student.objects.all().delete()
            self.stdout.write(f"Deleted {deleted} existing students.")
        try:
            result = import_dataframe(read_table(opts["file"]))
        except (ValueError, OSError) as exc:
            raise CommandError(str(exc))
        self.stdout.write(self.style.SUCCESS(
            f"Imported {result['created']} students, skipped {result['skipped']}."))
        for err in result["errors"]:
            self.stdout.write(self.style.WARNING(err))
        if not result["scored"]:
            self.stdout.write(self.style.WARNING("Not scored: run `python manage.py train_model`."))
