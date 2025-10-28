use aion_math::continuous_math::ContinuousMath;
use log::info;

use crate::node::Node;

pub struct Footstep {
    pub inputs: [f32; 6],
    pub logits: [f32; 3],
    pub probs: [f32; 3],
    pub action_total: [f32; 3],
    pub reward: f32,
    pub loss: f32,
    pub loss_average: f32,
    pub loss_trend: f32,
    pub adjusted_reward: f32,
    pub epsilon: f32,
    pub temperature: f32,
    pub action: usize,
    pub batch_number: usize,
    pub success: bool,
    pub diverging: bool,
    pub stuck: bool,
}

impl Footstep {
    pub fn new() -> Self {
        Self {
            inputs: [0.0; 6],
            logits: [0.0; 3],
            probs: [0.0; 3],
            action_total: [0.0; 3],
            reward: 0.0,
            loss: 0.0,
            loss_average: 0.0,
            loss_trend: 0.0,
            epsilon: 0.0,
            temperature: 0.0,
            adjusted_reward: 0.0,
            action: usize::MAX,
            batch_number: usize::MAX,
            success: false,
            diverging: false,
            stuck: false,
        }
    }

    pub fn print(
        &self,
        inputs: &[f32; 6],
        logits: [f32; 3],
        probs: [f32; 3],
        action: usize,
        reward: f32,
        success: bool,
        math: &mut ContinuousMath,
        adjusted_reward: f32,
        node: &Node
    ) {
        // TODO: Change the log using struct that provides inputs as vectors and custom log

        info!("Current features: CPU: {:.2}, MEM: {:.2}, SWAP: {:.2}, DISK: {:.2}, THROUGHPUT: {:.2}, LATENCY: {:.2}",
      self.  inputs[0],self. inputs[1],self. inputs[2],self. inputs[3],self. inputs[4],self. inputs[5]);

        info!("logits: {:?}", self.logits);
        info!("probs: {:?}", self.probs);
        info!(
            "Action taken: {}, Reward: {:.3}, adjusted_Reward: {:.3}, Success: {}. Diverging: {}, Stuck: {}",
         self.   action,
          self.  reward,
           self. adjusted_reward,
           self. success,
            self.diverging,
            self.stuck
        );

        let loss_avg = math.calc_avg_and_trend(1, self.loss, 0.005);
        info!(
            "epsilon: {:.3}, temperature: {:.3}, loss: {:.3}, loss_avg: {:.3}, loss_trend: {:.3}, batch: #{}",
            self.epsilon,
            self.temperature,
            self.loss,
            self.loss_average,
            self.loss_trend,
            self.batch_number
        );
        let action_details_guard = node.action_details;
        if action_details_guard.len() > 2 {
            let success_rates = node.get_success_rates();
            let totals = node.get_action_total();
            let totals_ratios = node.get_action_ratios();
            info!(
                "Total action: [High pressure] {}\t[Normal] {}\t[Low pressure] {}",
                totals[0], totals[1], totals[2]
            );
            info!(
                "Total ratios: [High pressure] {:.2}\t[Normal] {:.2}\t[Low pressure] {:.2}",
                totals_ratios[0], totals_ratios[1], totals_ratios[2]
            );
            info!(
                "Success Rate: [High pressure] {:.2}\t[Normal] {:.2}\t[Low pressure] {:.2}",
                success_rates[0], success_rates[1], success_rates[2]
            );
            info!(
                "Averages: [Total] {:.2}\t[Total ratio] {:.2}\t[Success rates] {:.2}",
                totals.iter().sum::<u32>() as f32 / totals.len() as f32,
                totals_ratios.iter().sum::<f32>() / totals_ratios.len() as f32,
                success_rates.iter().sum::<f32>() / success_rates.len() as f32
            );
            let reward_averages = vec![
                action_details_guard.get(&0).unwrap().reward / totals[0] as f32,
                action_details_guard.get(&1).unwrap().reward / totals[1] as f32,
                action_details_guard.get(&2).unwrap().reward / totals[2] as f32,
            ];
            info!(
                "Reward average: [High pressure] {:.2}\t[Normal] {:.2}\t[Low pressure] {:.2}",
                reward_averages[0], reward_averages[1], reward_averages[2]
            );
            info!(
                "[variance] Total: {:.3}, Success Rate: {:.3}, Reward averages: {:.3}, logits: {:.3}, probs: {:.3},",
                Math::variance_of_ratios(totals.iter().map(|x| *x as f32).collect()),
                Math::variance(success_rates),
                Math::variance_of_ratios(reward_averages),
                Math::variance(logits.to_vec()),
                Math::variance(probs.to_vec())
            );
        }
        println!("-----------------------------------------------------------------");
    }
}
