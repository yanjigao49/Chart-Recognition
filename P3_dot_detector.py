import os
import json
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.utils.data
import torch.optim as optim
from collections import OrderedDict, defaultdict
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import nms
from torchvision.models.detection.rpn import concat_box_prediction_layers

class DotDataset(torch.utils.data.Dataset):
    def __init__(self, root, num_samples=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "annotations"))))

        # Filter samples to ensure only dot charts
        self.imgs, self.annots = self.filter_samples(self.imgs, self.annots, num_samples)

    def filter_samples(self, imgs, annots, num_samples):
        dot_imgs = []
        dot_annots = []
        count = 0
        for img, annot in zip(imgs, annots):
            annot_path = os.path.join(self.root, "annotations", annot)
            with open(annot_path) as f:
                data = json.load(f)
                if data['chart-type'] == 'dot':
                    dot_imgs.append(img)
                    dot_annots.append(annot)
                    count += 1
                    if count >= num_samples:
                        break

        print(f"Dot samples: {len(dot_imgs)}")
        return dot_imgs, dot_annots

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "annotations", self.annots[idx])
        img = Image.open(img_path).convert("L")

        with open(annot_path) as f:
            annot = json.load(f)

        boxes = []
        labels = []
        for dot in annot['visual-elements']['dot points']:
            for point in dot:
                x = point['x']
                y = point['y']
                box_size = 25  # Define the size of the box around the dot
                x_min = x - box_size / 2
                y_min = y - box_size / 2
                x_max = x + box_size / 2
                y_max = y + box_size / 2
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  # Label for dots

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        target = {"boxes": boxes, "labels": labels, "image_id": image_id}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def transform(img, target):
    img = F.to_tensor(img)
    return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def evaluate_and_visualize(model, data_loader, device, iou_threshold=0.75, score_threshold=0.65):
    model.eval()
    cpu_device = torch.device("cpu")
    results = []
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        results.extend(outputs)

        for image, target, output in zip(images, targets, outputs):
            output = apply_nms(output, iou_threshold, score_threshold)
            visualize_predictions(image, target, output)

    return results


def apply_nms(output, iou_threshold, score_threshold):
    boxes = output['boxes']
    scores = output['scores']
    indices = nms(boxes, scores, iou_threshold)
    nms_boxes = boxes[indices]
    nms_scores = scores[indices]
    nms_labels = output['labels'][indices]

    # Filter out boxes with scores below threshold
    keep = nms_scores >= score_threshold
    nms_boxes = nms_boxes[keep]
    nms_scores = nms_scores[keep]
    nms_labels = nms_labels[keep]

    return {'boxes': nms_boxes, 'scores': nms_scores, 'labels': nms_labels}


def visualize_predictions(image, target, output, save_path=None):
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image, cmap='gray')

    # Plot ground truth boxes in green
    for box in target['boxes']:
        box = box.detach().cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='g',
                                 facecolor='none')
        ax.add_patch(rect)

    # Plot predicted boxes in red
    for box, score in zip(output['boxes'], output['scores']):
        box = box.detach().cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 10, f'{score:.2f}', color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    num_samples = 500  # Define the number of samples to use

    # Load the dot dataset
    dataset = DotDataset(root='train', num_samples=num_samples, transforms=transform)

    # Split the dataset into train, validation, and test sets
    def balanced_split(dataset, num_samples, val_ratio=0.1, test_ratio=0.1):
        indices = list(range(len(dataset)))
        val_size = int(num_samples * val_ratio)
        test_size = int(num_samples * test_ratio)
        train_size = num_samples - val_size - test_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return train_indices, val_indices, test_indices


    train_indices, val_indices, test_indices = balanced_split(dataset, num_samples)

    # Create subsets
    dataset_train = torch.utils.data.Subset(dataset, train_indices)
    dataset_val = torch.utils.data.Subset(dataset, val_indices)
    dataset_test = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4,
                                                    collate_fn=collate_fn)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=4,
                                                  collate_fn=collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4,
                                                   collate_fn=collate_fn)

    # Model and optimizer
    model = get_model(num_classes=2)  # 2 classes: background, dot
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)  # Use Adam optimizer


    def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
        model.train()
        running_loss = 0.0
        i = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            i += 1
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")

        return running_loss / len(data_loader)


    def eval_forward(model, images, targets):
        model.eval()
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = model.transform(images, targets)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        model.rpn.training = True

        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
        anchors = model.rpn.anchor_generator(images, features_rpn)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        proposal_losses = {}
        assert targets is not None
        labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        proposal_losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        image_shapes = images.image_sizes
        proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals,
                                                                                                      targets)
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)

        result = []
        detector_losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals,
                                                                       image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        detections = result
        detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        model.rpn.training = False
        model.roi_heads.training = False
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections


    def evaluate(model, data_loader, device):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                losses, _ = eval_forward(model, images, targets)
                val_loss += sum(loss.item() for loss in losses.values())

        return val_loss / len(data_loader)


    # Early stopping and model checkpoint
    class EarlyStopping:
        def __init__(self, patience=5, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float('inf')
            self.delta = delta

        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    pass
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            '''Saves model when validation loss decrease.'''
            if self.verbose:
                # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                pass
            torch.save(model.state_dict(), 'dot_detector_20.pth')
            self.val_loss_min = val_loss


    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, num_epochs)
        val_loss = evaluate(model, data_loader_val, device)
        print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # Call early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Training completed")

    # Evaluate the model and visualize predictions
    results = evaluate_and_visualize(model, data_loader_test, device)
