# --- Step 7: Training Pipeline (Modified to use references) ---
def train_harmonizer_pipeline(train_references, test_references,
                              base_directory, device, num_classes=2, batch_size=1, epochs=5):
    
    train_dataset = Variable4DDataset(train_references, base_directory)
    test_dataset = Variable4DDataset(test_references, base_directory)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = CNN_LSTM_Harmonizer(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train CNN+LSTM feature extractor
    print("Training CNN-LSTM feature extractor...")
    for epoch in range(epochs):
        print(epoch)
        model.train()
        total_loss = 0
        for i, (imgs, lbls, _) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} finished, Avg Loss: {total_loss / (i+1):.4f}")
        # Clear CUDA cache after each epoch to free up unused memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Extract features for harmonization
    print("Extracting features for harmonization...")
    train_feats, train_labels_out, train_sites_out = extract_features(model, train_loader, device)
    test_feats, test_labels_out, test_sites_out = extract_features(model, test_loader, device)
    
    # Learn and apply harmonization
    print("Learning and applying harmonization...")
    # Label encode site_ids for neuroHarmonize if they are strings
    site_encoder = LabelEncoder()
    train_sites_encoded = site_encoder.fit_transform(train_sites_out)
    test_sites_encoded = site_encoder.transform(test_sites_out) # Use transform for test set

    # neuroHarmonize expects 'covariates' as a pandas DataFrame or similar for the 'batches' column.
    # For a simple list of batches, it should be fine, but often it wants a specific structure.
    # Check neuroHarmonize documentation for 'batches' format. If it's a list, it should work.
    
    harmon_model = harmonizationLearn(train_feats, train_sites_encoded) # Assuming batches are site_ids
    train_feats_h = harmonizationApply(train_feats, harmon_model)
    test_feats_h = harmonizationApply(test_feats, harmon_model)
    
    # Fine-tune classifier head on harmonized features
    print("Fine-tuning classifier on harmonized features...")
    classifier = torch.nn.Linear(train_feats_h.shape[1], num_classes).to(device)
    clf_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    clf_criterion = torch.nn.CrossEntropyLoss()
    
    # Create DataLoader for harmonized features if they are too large to fit as single tensor
    # For now, assuming they fit, as features are much smaller than raw images.
    X_train = torch.tensor(train_feats_h, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels_out, dtype=torch.long).to(device)
    X_test = torch.tensor(test_feats_h, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels_out, dtype=torch.long).to(device)
    
    for epoch_clf in range(20):
        classifier.train()
        clf_optimizer.zero_grad()
        logits = classifier(X_train)
        loss = clf_criterion(logits, y_train)
        loss.backward()
        clf_optimizer.step()
        if (epoch_clf + 1) % 5 == 0:
            print(f"  Classifier Epoch {epoch_clf+1}/20, Loss: {loss.item():.4f}")
    
    classifier.eval()
    with torch.no_grad():
        preds = torch.argmax(classifier(X_test), dim=1).cpu().numpy()
        acc = accuracy_score(y_test.cpu().numpy(), preds)
    
    print(f"Final test accuracy after harmonization: {acc*100:.2f}%")
    
    return model, harmon_model, classifier
