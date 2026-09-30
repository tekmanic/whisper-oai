package server

import (
	"crypto/subtle"
	"net/http"
)

// authMiddleware returns a middleware that, when apiKey != "", requires
// "Authorization: Bearer <apiKey>" (401 otherwise). When apiKey == "", it is a
// no-op passthrough.
func authMiddleware(apiKey string, next http.Handler) http.Handler {
	if apiKey == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		const prefix = "Bearer "
		header := r.Header.Get("Authorization")
		ok := false
		if len(header) > len(prefix) &&
			subtle.ConstantTimeCompare([]byte(header[:len(prefix)]), []byte(prefix)) == 1 {
			ok = subtle.ConstantTimeCompare([]byte(header[len(prefix):]), []byte(apiKey)) == 1
		}
		if ok {
			next.ServeHTTP(w, r)
			return
		}
		writeError(w, http.StatusUnauthorized, "authentication_error", "Invalid API key", "")
	})
}
