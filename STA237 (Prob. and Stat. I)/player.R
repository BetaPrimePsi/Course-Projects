player <- function(player_cards, dealer_card) {
    score <- score_blackjack(player_cards)
    hasace <- 'A' %in% player_cards$numbers
    deal <- sum(dealer_card$value)
    decision <- TRUE
    if (score == 12 & ((2 <= deal & deal <= 3) | deal >= 7)) {
        decision <- TRUE
    }
    if (score == 12 & (4 <= deal & deal <= 6)) {
        decision <- hasace
    }
    if ((13<= score & score <= 16) & deal <= 6) {
        decision <- hasace
    }
    if ((13<= score & score <= 16) & deal >= 7) {
        decision <- TRUE
    }
    if (score==17) {
        decision <- hasace
    }
    if (score==18 & deal <= 8) {
        decision <- FALSE
    }
    if (score >= 19) {
        decision <- FALSE
    }
    
  return(decision)
}