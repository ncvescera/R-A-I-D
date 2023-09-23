import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch import nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchview import draw_graph
import graphviz
import wandb
import plotly.express as px
from torchmetrics.regression import MeanAbsolutePercentageError, SymmetricMeanAbsolutePercentageError, MeanAbsoluteError, R2Score
from IPython.display import display, HTML
from .preprocessing import CustomDataset, collate_fn


graphviz.set_jupyter_format('png')


class ModelManager():
    def __init__(self, run, model_params: dict = {}, optimizer="adam", name="custommodel",bsize=10, forcecpu=False):
        self.BATCH_SIZE = bsize
        self.forcecpu = forcecpu
        self.run = run
        self.name = name
        
        self.device = self.__init_device()
        
        # feature_size=48, future_size=18, gru_out=32, decoder_input=128
        self.model = NeuralNetwork(**model_params).to(self.device)
        
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.run.config['learning_rate'])
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.run.config['learning_rate'])
        else:
            raise ValueError(" optimizer adam o sgd")

        print(self.model)
        print(self.optimizer)

        self.run.config['optimizer'] = self.optimizer
        
        self.__model_graph()
        
    def __init_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        if self.forcecpu:
            device = "cpu"

        print(f"Using {device} device")

        return device

    def __model_graph(self):
        model_graph = draw_graph(
            self.model, 
            input_size=[(5, 287, 48), (5, 287, 48), (5, 287, 18)],
            save_graph=True, expand_nested=True, device=self.device
        )
        model_graph.visual_graph

        self.run.log({"Model Shape": wandb.Image("model.gv.png")})
    
    def train(self, train_df, val_df, ccollate_fn=None):
        # definizione datasets e dataloaders
        # vengono istanziati ogni volta che la run ha inizio così da gestire meglio ed impedire
        # errori sul calcolo dei batch
        train_dataset = CustomDataset(train_df, self.BATCH_SIZE)
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
        val_dataset = CustomDataset(val_df, self.BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

        t_loss_list, t_loss_notnorm_list, v_loss_list, v_loss_notnorm_list = [], [], [], []
        loop = tqdm(range(self.run.config['epochs']), unit="epoch")
        early_stopping = EarlyStopper(patience=20)
        best_vloss = 1_000_000.
        loss = nn.L1Loss()

        self.run.config['early_stopping'] = str(early_stopping)
        self.run.config['save_best'] = True
        self.run.watch(self.model)

        for epoch in loop:
            self.model.train()
            for batch_idx, (before, target, future, after, m_b, b_t, m_a, timestamps, _) in enumerate(train_dataloader):        
                target, before, after, future = (target.to(self.device), before.to(self.device), 
                                                    after.to(self.device), future.to(self.device))

                train_pred = self.model(before, after, future)
                train_loss = loss((train_pred/train_pred.sum(dim=(1,2), keepdim=True)) * target.sum(dim=(1,2), keepdim=True), target)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                t_loss_list.append(train_loss.item())

                with torch.no_grad(): # TODO: ricontrollare
                    t_loss_notnorm_list.append(loss(train_pred, target).item())

            self.model.eval()
            running_vloss = 0.0
            running_vloss_notnorm = 0.0

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, (vbefore, vtarget, vfuture, vafter, vm_b, vm_t, vm_a, vtimestamp, _) in enumerate(val_dataloader):
                    vbefore, vtarget, vfuture, vafter = (vbefore.to(self.device), vtarget.to(self.device),
                                                            vfuture.to(self.device), vafter.to(self.device))
                    
                    voutputs = self.model(vbefore, vafter, vfuture)
                    
                    vloss = loss((voutputs/voutputs.sum(dim=(1,2), keepdim=True)) * vtarget.sum(dim=(1,2), keepdim=True), vtarget)
                    vloss_notnorm = loss(voutputs, vtarget)

                    running_vloss += vloss
                    running_vloss_notnorm += vloss_notnorm

            # resetting dataset generation
            # TODO: fix con DROP LAST
            train_dataset.reset()
            val_dataset.reset()

            avg_vloss = running_vloss / (i + 1) # TODO: ricontrollare
            avg_vloss_notnorm = running_vloss_notnorm / (i + 1) # TODO: ricontrollare

            v_loss_list.append(avg_vloss.item())
            v_loss_notnorm_list.append(avg_vloss_notnorm.item())

            loop.set_description(f"Epoch [{epoch}/{self.run.config['epochs']}]")
            loop.set_postfix(loss=t_loss_list[-1], val_loss=v_loss_list[-1])

            self.run.log({"loss": t_loss_list[-1], "val_loss": v_loss_list[-1]})

            print(
                    f"Epoch {epoch+1:02d}/{self.run.config['epochs']:02d}"
                    f" | Train Loss {t_loss_list[-1]:.3f}"
                    f" | Val Loss {v_loss_list[-1]:.3f}"
                )

            # save model when improves loss
            if avg_vloss < best_vloss:
                print(f"\tSaving model (improve loss) {best_vloss} -> {avg_vloss}")
                best_vloss = avg_vloss
                self.model_path = f"./models/{self.name}_b{self.run.config['batch_size']}_e{self.run.config['epochs']}.model"
                torch.save(self.model.state_dict(), self.model_path)

            # check for early stopping
            if early_stopping.early_stop(v_loss_list[-1]):
                print(f"\tEarly Stopping @ epoch {epoch} loss: {t_loss_list[-1]} val_loss: {v_loss_list[-1]}")
                break

        # plot training/validation loss
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))
        axes[0, 0].set_title("Train Loss (Norm)")
        axes[0, 0].plot(t_loss_list)
        axes[0, 1].set_title("Val Loss (Norm)")
        axes[0, 1].plot(v_loss_list)
        axes[1, 0].set_title("Train Loss (Not Norm)")
        axes[1, 0].plot(t_loss_notnorm_list)
        axes[1, 1].set_title("Val Loss (Not Norm)")
        axes[1, 1].plot(v_loss_notnorm_list)
        fig.tight_layout()
        
    def evaluate(self, test_df, train_target_scaler, model_path:str=None, finish_run=True):
        # pd.set_option("display.precision", 2) # TODO: forse va rimosso
        
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
        test_dataset = CustomDataset(test_df, 1)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False
        )

        result = pd.DataFrame()
        mean_abs_percentage_error = MeanAbsolutePercentageError()
        smape = SymmetricMeanAbsolutePercentageError()
        r2 = R2Score()
        mae = MeanAbsoluteError()
        nrmse = self.NormalizedRootMeanSquareError
        mape_points = []
        mape = 0

        # Create a table
        table = wandb.Table(columns = ["Evaluation (on Testing Set)", "MAPE(+)", "SMAPE", "NRMSE", "MAE", "R2"])

        # Create path for Plotly figure
        path_to_plotly_html = "./plotly_figure.html"

        for i, (vbefore, vtarget, vfuture, vafter, vm_b, vm_t, vm_a, timestamps, solargis) in enumerate(test_dataloader):
            timestamp =  pd.to_datetime(pd.Series(timestamps[0]))
            timestamp = timestamp[vm_t[0] == 1]

            vbefore, vafter, vfuture = vbefore.to(self.device), vafter.to(self.device), vfuture.to(self.device)
            prediction = self.model(vbefore, vafter, vfuture)
            prediction = prediction.cpu().detach()
            # prediction = prediction * vfuture.cpu().detach()[:,:,-1].reshape(prediction.shape) # output * isday
            prediction = (prediction/prediction.sum(dim=(1,2), keepdim=True)) * vtarget.sum(dim=(1,2), keepdim=True)

            # tutto quello che è < 0 diventa 0
            prediction[prediction < 0] = 0

            # TODO: ricontrollare mape
            #step_mape = mean_abs_percentage_error(target=vtarget, preds=prediction).item()
            step_mape_positives = mean_abs_percentage_error(target=vtarget[vtarget>0], preds=prediction[vtarget>0])
            step_smape = smape(target=vtarget, preds=prediction)
            step_mae = mae(target=vtarget, preds=prediction)
            step_r2 = r2(target=vtarget.squeeze(0), preds=prediction.squeeze(0))
            step_nrmse = nrmse(target=vtarget, preds=prediction)

            ndays = int(np.count_nonzero(vm_t[0]==1)/test_dataset.DAY_IN_TIMESTAMPS)

            step_dailymae = self.daily_mae(vtarget, prediction, ndays, test_dataset.DAY_IN_TIMESTAMPS)

            #vtarget = vtarget.numpy().reshape(vtarget.shape[1], 1)
            #prediction = prediction.numpy().reshape(prediction.shape[1],1)
            #step_mape_sklearn = mean_absolute_percentage_error(vtarget[vtarget>0], prediction[vtarget>0])

            mape_points.append(step_mape_positives)
            mape += step_mape_positives

            vtarget = pd.DataFrame(vtarget.numpy().reshape(vtarget.shape[1], 1))
            #vtarget = pd.DataFrame(vtarget)
            vtarget.columns = ['target']
            # target non normalizzata e cumulata
            unnormalized_target = pd.DataFrame((train_target_scaler.inverse_transform(vtarget)).cumsum())
            unnormalized_target.columns = ['target (real)']
            # target non normalizzata con differenze
            unnormalized_target_diff = pd.DataFrame(train_target_scaler.inverse_transform(vtarget))
            unnormalized_target_diff.columns = ['target (real)']

            prediction = pd.DataFrame(prediction.numpy().reshape(prediction.shape[1],1))
            #prediction = pd.DataFrame(prediction)
            prediction.columns = ['prediction']

            # preds non normalizzata e cumulata
            unnormalized_prediction = pd.DataFrame((train_target_scaler.inverse_transform(prediction)).cumsum())
            unnormalized_prediction.columns = ['prediction (real)']
            # preds non normalizzata con differenze
            unnormalized_prediction_diff = pd.DataFrame(train_target_scaler.inverse_transform(prediction))
            unnormalized_prediction_diff.columns = ['prediction (real)']

            solargis = solargis[0]

            step_result = pd.concat([vtarget, prediction], axis=1)
            step_result.index = timestamp
            # non normalizzata e cumsum
            step_result_real = pd.concat([unnormalized_target, unnormalized_prediction], axis=1)
            step_result_real.index = timestamp
            # non normalizzata con differenze
            step_result_real_diff = pd.concat([unnormalized_target_diff, unnormalized_prediction_diff], axis=1)
            step_result_real_diff.index = timestamp

            step_result = pd.concat([step_result, solargis], axis=1)

            fig = px.line(step_result, y=['target', 'prediction', 'GHI', 'GTI'], title=f"Step Prediction ({timestamp.dt.day.values[0]}-{timestamp.dt.month.values[0]}, {int(np.count_nonzero(vm_t[0]==1)/test_dataset.DAY_IN_TIMESTAMPS)} d)", markers=True)
            fig.write_html(path_to_plotly_html, auto_play=False) 
            fig.show()

            fig = px.line(step_result_real_diff, y=['target (real)', 'prediction (real)'], title=f"Step Real Diff Prediction ({timestamp.dt.day.values[0]}-{timestamp.dt.month.values[0]}, {int(np.count_nonzero(vm_t[0]==1)/test_dataset.DAY_IN_TIMESTAMPS)} d)", markers=True)
            fig.show()

            fig = px.line(step_result_real, y=['target (real)', 'prediction (real)'], title=f"Step CumSum Prediction ({timestamp.dt.day.values[0]}-{timestamp.dt.month.values[0]}, {int(np.count_nonzero(vm_t[0]==1)/test_dataset.DAY_IN_TIMESTAMPS)} d)", markers=True)
            fig.show()

            table.add_data(wandb.Html(path_to_plotly_html), step_mape_positives, step_smape, step_nrmse, step_mae, step_r2)

            result =  pd.concat([result, step_result])

            step_realmae = mae(target=torch.tensor(step_result_real['target (real)'].values), preds=torch.tensor(step_result_real['prediction (real)'].values)).item()
            step_realmape = mean_abs_percentage_error(target=torch.tensor(step_result_real['target (real)'].values), preds=torch.tensor(step_result_real['prediction (real)'].values)).item()
            step_realsmape = smape(target=torch.tensor(step_result_real['target (real)'].values), preds=torch.tensor(step_result_real['prediction (real)'].values)).item()
            step_realr2 = r2(target=torch.tensor(step_result_real['target (real)'].values), preds=torch.tensor(step_result_real['prediction (real)'].values))

            step_real_dailymae = self.daily_mae(
                torch.tensor(step_result_real['target (real)'].to_numpy().reshape(1, step_result_real['target (real)'].shape[0], 1)), 
                torch.tensor(step_result_real['prediction (real)'].to_numpy().reshape(1, step_result_real['prediction (real)'].shape[0], 1)), ndays, test_dataset.DAY_IN_TIMESTAMPS)
            step_real_dailymae.columns = ['MAE (kWh)', 'MAPE (%)']


            step_realmae_diff = mae(target=torch.tensor(step_result_real_diff['target (real)'].values), preds=torch.tensor(step_result_real_diff['prediction (real)'].values)).item()
            step_realmape_diff = mean_abs_percentage_error(target=torch.tensor(step_result_real_diff['target (real)'].values), preds=torch.tensor(step_result_real_diff['prediction (real)'].values)).item()
            step_realsmape_diff = smape(target=torch.tensor(step_result_real_diff['target (real)'].values), preds=torch.tensor(step_result_real_diff['prediction (real)'].values)).item()
            step_realr2_diff = r2(target=torch.tensor(step_result_real_diff['target (real)'].values), preds=torch.tensor(step_result_real_diff['prediction (real)'].values))

            step_real_dailymae_diff = self.daily_mae(
                torch.tensor(step_result_real_diff['target (real)'].to_numpy().reshape(1, step_result_real_diff['target (real)'].shape[0], 1)), 
                torch.tensor(step_result_real_diff['prediction (real)'].to_numpy().reshape(1, step_result_real_diff['prediction (real)'].shape[0], 1)), ndays, test_dataset.DAY_IN_TIMESTAMPS)
            step_real_dailymae_diff.columns = ['MAE (kWh)', 'MAPE (%)']

            normdata_plot= pd.DataFrame(
                [step_mape_positives*100,step_smape*100,step_nrmse*100,step_mae,step_r2], columns=['Value'], index=['MAPE + (%)', 'SMAPE (%)', 'NRMSE (%)', 'MAE ()', 'R2'],
                dtype=float
            )
            normdata_plot = normdata_plot.astype(float).round(decimals=2)
            realdata_plot = pd.DataFrame(
                [step_realmae], index=['MAE (kWh)'], columns=['Value'],
                dtype=float
            )
            realdata_plot=realdata_plot.astype(float).round(decimals=2)

            realdata_diff_plot = pd.DataFrame(
                #[step_realmape_diff*100,step_realsmape_diff*100,step_realmae_diff,step_realr2_diff], columns=['Value'], index=['MAPE (%)', 'SMAPE (%)', 'MAE ()', 'R2'],
                [step_realmae_diff], columns=['Value'], index=['MAE (kW/h)'],
                dtype=float
            )
            realdata_diff_plot=realdata_diff_plot.astype(float).round(decimals=2)

            # plot Dataframes side by side
            html_str = (
                normdata_plot.style.set_table_attributes("style='display:inline; margin:3em;'").set_caption('<h3>Normalized Data Results</h3>').render() +
                step_dailymae.style.set_table_attributes("style='display:inline; margin:3em;'").set_caption('<h3>Normalized Daily MAE Results</h3>').render()+
                realdata_diff_plot.style.set_table_attributes("style='display:inline; margin:3em;'").set_caption('<h3>Real Data Results</h3>').render()+
                step_real_dailymae_diff.style.set_table_attributes("style='display:inline; margin:3em;'").set_caption('<h3>Real Daily MAE Results</h3>').render()

            )
            display(HTML(html_str))

        # Log Table
        self.run.log({"evluation_table": table})

        if finish_run:
            self.run.finish()
    
    def __find_lr(self):
        raise NotImplementedError()
        
    def NormalizedRootMeanSquareError(self,target, preds):
        
        def RootMeanSquareError():
            return np.sqrt((1/target.shape[1]) * torch.pow((target-preds), 2).sum())

        RANGE = torch.max(target)-torch.min(target)

        return RootMeanSquareError() / RANGE
    
    def daily_mae(self, target:torch.Tensor, preds:torch.Tensor, days:int, day_in_timestamp: int):
        start = 0
        result = {}
        for day in range(days):
            t = pd.DataFrame(target.numpy().reshape(target.shape[1], 1))
            p = pd.DataFrame(preds.numpy().reshape(preds.shape[1], 1))

            t = t.iloc[start:start+day_in_timestamp]
            p = p.iloc[start:start+day_in_timestamp]

            start += day_in_timestamp

            # TODO: ricontrollare
            step_mae = float(abs(t.sum() - p.sum()))
            step_pmae = float(step_mae / t.sum()) * 100.0

            result[f'Day {day+1}'] = [step_mae, step_pmae]

        result = pd.DataFrame(result).T
        result.columns = ['MAE ()', "MAPE (%)"]

        # display(result)
        return result
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def __str__(self):
        return f"EarlyStopper(patience: {self.patience}, min_delta={self.min_delta})"

    
class NeuralNetwork(nn.Module):
    # TODO: passare questi dati come parametri al costruttore
    #GRU_OUT_SIZE = 32
    ##BATCH_SIZE = run.config['batch_size']
    #FEATURES_SIZE = 48
    #DECODER_INPUT_SIZE = 128
    #FUTURE_SIZE = 18
    
    def __init__(self, feature_size=48, future_size=18, gru_out=32, decoder_input=128, bsize=10):
        super().__init__()
        
        self.BATCH_SIZE = bsize
        
        # ENCODER
        self.FEATURES_SIZE = feature_size
        self.FUTURE_SIZE = future_size
        self.GRU_OUT_SIZE = gru_out
        self.DECODER_INPUT_SIZE = decoder_input
        
        self.input_before = nn.GRU(
            input_size=self.FEATURES_SIZE, 
            hidden_size=self.GRU_OUT_SIZE, 
            num_layers=1, 
            batch_first=True
        )# fare prove. usare multipli di 2 forse meglio ??
        self.input_after = nn.GRU(
            input_size=self.FEATURES_SIZE,
            hidden_size=self.GRU_OUT_SIZE,
            num_layers=1,
            batch_first=True
        )
        
        #FCL 
        self.flat = nn.Flatten()
        
        self.linear1_2 = nn.Linear(self.GRU_OUT_SIZE * 2, self.DECODER_INPUT_SIZE)
        self.linear2_2 = nn.Linear(self.DECODER_INPUT_SIZE, self.DECODER_INPUT_SIZE)
        
        # DECODER
        self.loopGru = nn.GRU(
            input_size=self.FUTURE_SIZE, 
            hidden_size=self.DECODER_INPUT_SIZE, 
            num_layers=1, 
            batch_first=True
        )
        
        # porta a [len_buco, 1] l'output della gru
        self.output_layer = nn.Linear(self.DECODER_INPUT_SIZE, 1)

    def encoder(self, before, after):
        before_out, before_h = self.input_before(before)
        after_out, after_h = self.input_after(after)
              
        # prendere l'ultima predizione della GRU x
        before_out = before_out[:, -1:]
        after_out  = after_out[:, -1:]

        #before_h   = before_h[:, -1:]
        #after_h    = after_h[:, -1:]
        
        # combina le features
        x = torch.cat((before_out, after_out), -1)
        # hidden_state = torch.cat((before_h, after_h), -1)
        
        return x, None
    
    def middle_layer(self, x):
        x = F.relu(self.linear1_2(x))
        x = F.relu(self.linear2_2(x))
        
        return x
    
    def decoder(self, future_input, hidden_state):
        # effettuo uno swap tra la batch_size e la dimensione del buco
        # per rispettare l'input dell'hidden state
        hidden_state = hidden_state.permute(1, 0, 2)
        
        x, hidden = self.loopGru(future_input, hidden_state)
        
        return x
    
    def output(self, x, future):
        x = self.output_layer(x)
        
        # output * isday
        # future[:,:,-1] prende l'ultima feaure di future che è isday
        x = (x * future[:,:,-1].reshape(x.shape))
        
        # TODO: normalizzazione qui !
        # x = ...
        
        return x

    def test(self, before, after, future_input):        
        x, hidden_state = self.encoder(before, after)
        print("Out Encoder: ", x.shape)
        
        x = self.middle_layer(x)
        print("Out Middle: ", x.shape)
        
        x = self.decoder(future_input, x)
        print("Out Decoder: ", x.shape)
        
        x = self.output(x)
        print("Output: ", x.shape)
        
        return x

    def forward(self, before, after, future_input):        
        x, _ = self.encoder(before, after)        
        x    = self.middle_layer(x)        
        x    = self.decoder(future_input, x)        
        x    = self.output(x, future_input)

        return x
