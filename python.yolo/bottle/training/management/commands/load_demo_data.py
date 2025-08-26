from django.core.management.base import BaseCommand
from training.models import TrainingSession, TrainingMetric, ClassMetric


class Command(BaseCommand):
    help = "Load demo data for YOLO training dashboard"

    def handle(self, *args, **options):
        # Create demo training session
        session, created = TrainingSession.objects.get_or_create(
            model_name="YOLOv8n",
            version="1.0.2",
            defaults={
                "status": "training",
                "dataset_name": "COCO 2017",
                "gpu_info": "RTX 4090",
                "memory_info": "24GB",
                "total_epochs": 100,
                "current_epoch": 50,
            },
        )

        if created:
            self.stdout.write(f"Created training session: {session}")
        else:
            self.stdout.write(f"Training session already exists: {session}")

        # Create demo training metrics
        demo_data = [
            (1, 0.8, 0.75, 0.42, 0.24),
            (5, 0.62, 0.58, 0.56, 0.32),
            (10, 0.48, 0.45, 0.68, 0.41),
            (15, 0.39, 0.37, 0.74, 0.47),
            (20, 0.34, 0.32, 0.78, 0.52),
            (25, 0.31, 0.29, 0.81, 0.55),
            (30, 0.28, 0.27, 0.83, 0.57),
            (35, 0.26, 0.25, 0.85, 0.59),
            (40, 0.24, 0.24, 0.86, 0.61),
            (45, 0.23, 0.23, 0.87, 0.62),
            (50, 0.22, 0.22, 0.88, 0.63),
        ]

        for epoch, train_loss, val_loss, map50, map95 in demo_data:
            metric, created = TrainingMetric.objects.get_or_create(
                session=session,
                epoch=epoch,
                defaults={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "map50": map50,
                    "map95": map95,
                    "precision": map50 * 0.9,
                    "recall": map50 * 0.95,
                },
            )
            if created:
                self.stdout.write(f"Created metric for epoch {epoch}")

        # Create demo class metrics
        class_data = [
            ("person", 0.89, 0.91, 0.90, 1247),
            ("car", 0.85, 0.87, 0.86, 892),
            ("bicycle", 0.78, 0.82, 0.80, 456),
            ("motorbike", 0.82, 0.79, 0.80, 234),
            ("bus", 0.91, 0.88, 0.89, 156),
            ("truck", 0.87, 0.85, 0.86, 203),
        ]

        for class_name, precision, recall, f1_score, instances in class_data:
            class_metric, created = ClassMetric.objects.get_or_create(
                session=session,
                class_name=class_name,
                defaults={
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "instances": instances,
                },
            )
            if created:
                self.stdout.write(f"Created class metric for {class_name}")

        self.stdout.write(self.style.SUCCESS("Successfully loaded demo data!"))
