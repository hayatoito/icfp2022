use crate::prelude::*;

// task/solution/
//   {id}-{score}.txt
// task/paint/
//   {id}.pgn
// task/best/
//   {id}.txt

#[derive(Serialize, Deserialize, Debug)]
pub struct Results {
    pub results: Vec<Submission>,
}

impl Results {
    pub fn best(&self, id: ProblemId) -> Option<Cost> {
        if let Some(min_cost) = self
            .results
            .iter()
            .find(|submission| submission.problem_id == id)
            .map(|s| s.min_cost)
        {
            // Judge return 0 if there is no submission.
            if min_cost == 0 {
                None
            } else {
                Some(min_cost)
            }
        } else {
            None
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Submission {
    pub problem_id: ProblemId,
    pub min_cost: Cost,
}

impl Results {
    pub fn new() -> Result<Results> {
        Ok(serde_json::from_str(&read_from_task_dir("result.json")?)?)
    }
}
