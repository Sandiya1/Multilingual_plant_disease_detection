from django.db import models

class Disease(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    cure = models.TextField()

    def __str__(self):
        return self.name

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    detected_disease = models.ForeignKey(Disease, on_delete=models.SET_NULL, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
