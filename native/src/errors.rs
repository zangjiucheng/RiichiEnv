use std::fmt;

#[derive(Debug)]
pub struct RiichiError {
    pub message: String,
}

impl RiichiError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl fmt::Display for RiichiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RiichiError {}

pub type RiichiResult<T> = Result<T, RiichiError>;

#[cfg(feature = "python")]
impl From<RiichiError> for pyo3::PyErr {
    fn from(err: RiichiError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(err.message)
    }
}
