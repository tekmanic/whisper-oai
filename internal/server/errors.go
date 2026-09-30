package server

import (
	"encoding/json"
	"net/http"
)

// errorDetail is the OpenAI error object. Param is a pointer so an empty
// param is omitted entirely; Code is always nil (null) and omitted via
// omitempty.
type errorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param,omitempty"`
	Code    *string `json:"code,omitempty"`
}

// errorBody is the OpenAI error envelope.
type errorBody struct {
	Error errorDetail `json:"error"`
}

// writeError writes the OpenAI error shape with the given HTTP status.
// errType is one of: invalid_request_error, authentication_error, server_error,
// service_unavailable. param may be "".
func writeError(w http.ResponseWriter, status int, errType, message, param string) {
	detail := errorDetail{Message: message, Type: errType}
	if param != "" {
		detail.Param = &param
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorBody{Error: detail})
}
