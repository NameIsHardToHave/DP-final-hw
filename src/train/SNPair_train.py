import torch
from tqdm import tqdm

# 训练循环
def train(train_dataloader, model, metricses, optimizer, logger, device):
    # 开启训练模式
    model.train()
    optimizer.zero_grad()
    los = 0
    n = len(train_dataloader)
    
    # 前向计算和反向传播
    for X, y in tqdm(train_dataloader):
        X, y = X.to(device), y.to(device)
        loss = model.forward_and_loss(X, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        los += loss.item()
    
    # 打印和收集信息
    los = los / n
    logger.info(f"Train Loss: {los:>7f}")
    mets = [0.0]
    return los, mets

# 验证循环
def val(val_dataloader, model, metricses, logger, device):
    # 开启验证模式
    model.eval()
    for metrics in metricses:
        metrics.reset()
    los = 0
    n = len(val_dataloader)
    
    # 前向计算
    with torch.no_grad():
        for X, y in tqdm(val_dataloader):
            X, y = X.to(device), y.to(device)
            loss = model.forward_and_loss(X, y)
            logits = model.inference(X)
            for metrics in metricses:
                metrics.update(logits, y)
            los += loss.item()
    
    # 打印和收集信息
    los = los / n
    logger.info(f"Val Loss: {los:>7f}")
    mets = []
    for i, metrics in enumerate(metricses):
        met = metrics.compute().item()
        logger.info(f"Val {metrics}: {met:>6f}")
        mets.append(met)
    return los, mets

# 完整的训练流程
def full_training(model, train_dataloader, val_dataloader, metricses, optimizer, logger, config):
    # 固定随机数种子
    torch.manual_seed(config.train.seed)
    
    # 开始训练
    logger.info("Training start!")
    
    # 定义损失函数/评估指标/优化器
    for metrics in metricses:
        metrics = metrics.to(config.device)

    # 准备迭代训练
    count = 0
    best_val_metr = 0
    torch.save(model.state_dict(), config.model.path+config.model.name)
    train_losses = []
    val_losses = []
    train_metrses = []
    val_metrses = []

    for t in range(config.train.epochs):
        # 开始训练/验证
        logger.info(f"Epoch {t+1} -------------------------------")
        train_loss, train_metrs = train(train_dataloader, model, metricses, optimizer, logger, config.device)
        val_loss, val_metrs = val(val_dataloader, model, metricses, logger, config.device)
        
        # 保存训练过程
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrses.append(train_metrs)
        val_metrses.append(val_metrs)

        # 验证集第一个指标不上升时早停，并根据验证集指标保留最佳参数
        if sum(val_metrs) > best_val_metr:
            best_val_metr = sum(val_metrs)
            torch.save(model.state_dict(), config.model.path+config.model.name)
            count = 0
        else:
            count += 1
        
        if count >= config.train.patience:
            logger.info(f'Early stopping at epoch {t}, best validation metrics sum: {best_val_metr}')
            break

    # 在早停后，恢复到最佳模型状态
    model.load_state_dict(torch.load(config.model.path+config.model.name))
    logger.info("Training Done!")
    
    # 打包训练信息
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_metricses': train_metrses,
        'val_metricses': val_metrses,
        'iters': t+1,
    }

# 完整的测试流程
def full_testing(model, test_dataloader, metricses, logger, config):
    # 开启测试模式
    model.eval()
    for metrics in metricses:
        metrics.reset()
    los = 0
    n = len(test_dataloader)
    logger.info("Testing Start!")
    
    # 前向计算
    y_preds = []
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            X, y = X.to(config.device), y.to(config.device)
            loss = model.forward_and_loss(X, y)
            logits = model.inference(X)
            y_pred = logits.argmax(dim=1)
            y_preds.append(y_pred)
            for metrics in metricses:
                metrics.update(logits, y)
            los += loss.item()
    y_preding = torch.cat(y_preds)

    # 打印和收集信息
    los = los / n
    logger.info(f"Test Loss: {los:>7f}")
    mets = []
    for i, metrics in enumerate(metricses):
        met = metrics.compute().item()
        logger.info(f"Test {metrics}: {met:>6f}")
        mets.append(met)
    logger.info("Testing Done!")
    
    return {
        'test_loss': los,
        'test_metrics': met,
        'y_pred': y_preding.cpu(),
    }
    
# 完整的预测流程
def full_predicting(model, test_dataloader, nums, config):
    # 开启测试模式
    model.eval()

    # 前向计算
    all_topk_preds = []
    with torch.no_grad():
        for X in tqdm(test_dataloader):
            X = X.to(config.device)
            output = model.inference(X)
            # 获取每一行的前10大元素的索引
            _, topk_indices = output.topk(nums, dim=1)
            all_topk_preds.append(topk_indices)

    # 得到最终结果
    y_topk_pred = torch.cat(all_topk_preds, dim=0)

    return y_topk_pred.cpu()